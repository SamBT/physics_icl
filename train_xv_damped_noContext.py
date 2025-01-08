import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import math
import numpy as np
from tqdm import tqdm

from model import GPT, GPTConfig
#from physics_data import RandomSHODataset, RandomSHODatasetVariableLength, RandomSHODatasetVariableLengthSmoothMass
from PhysicsDatasets import SHODataset, SHODatasetXV, DampedSHODatasetXV
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import os
import json

def main():
    has_cuda = torch.cuda.is_available()
    has_mps = torch.backends.mps.is_available()
    if has_cuda:
        device = 'cuda'
    elif has_mps:
        device = 'mps'
    else:
        device = 'cpu'
    device_type = device # for later use in torch.autocast

    # high-level
    now = datetime.now()
    formatted_date_time = now.strftime("%d%b%y_%H%M")
    model_name = f"random_DampedSHO_xvInput_noContext_{formatted_date_time}"

    outdir = f"trainings/{model_name}"
    logdir = f"logs/{model_name}"
    if not os.path.isdir(outdir):
        os.mkdir(outdir)
    if not os.path.isdir(logdir):
        os.mkdir(logdir)


    # masses for SHO dataset
    #masses = torch.cat([torch.arange(0.1,4.1,0.1),torch.arange(5.0,10.1,0.1)],dim=0)
    masses = [(1,4),(5,10)]
    masses_write = masses.tolist() if type(masses) == torch.Tensor else masses
    k = None
    beta = None
    dt = 0.1
    pin_amplitude = None
    min_amplitude = 0.0
    k_context = False
    context_dim = None
    min_seq_length = 10
    max_seq_length = 50 # so that we range up to 2pi with dt = 0.1 and k=10 -- m = 10 corresponds to T = 2pi = 65*0.1
    vary_length = True # whether or not to train with variable-length sequences

    # model parameters
    model_cfg = dict(
        block_size=1024, # maximum sequence length?
        input_dim=2, # 2d (x,v) time-series data
        context_dim=context_dim, # dimension of context vector (mass)
        n_layer=6,
        n_head=8,
        n_embd=256, # embedding dimensions
        dropout=0.0,
        bias=False,
        use_pe=False,
        use_rope=True
    )

    # train/val info
    num_train = 20_000
    num_val = 10_000
    bs = 256
    bs_val = 1024
    num_epoch = 100
    max_iters = num_epoch*(num_train/bs)

    # optimizer
    learning_rate = 5e-4
    weight_decay = 0.01
    beta1 = 0.9
    beta2 = 0.95
    grad_clip = 1.0

    # learning rate schedule
    decay_lr = True # whether to decay the learning rate
    warmup_iters = 0.05*max_iters # how many steps to warm up for
    lr_decay_iters = max_iters # should be ~= max_iters per Chinchilla
    min_lr = 1e-6 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla

    training_cfg = dict(
        model_name=model_name,
        masses=masses_write,
        k=k,
        beta=beta,
        pin_amplitude=pin_amplitude,
        min_amplitude=min_amplitude,
        k_context=k_context,
        num_train=num_train,
        num_val=num_val,
        min_seq_length=min_seq_length,
        max_seq_length=max_seq_length,
        vary_length=vary_length,
        dt=dt,
        bs=bs,
        bs_val=bs_val,
        num_epoch=num_epoch,
        max_iters=max_iters,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        beta1=beta1,
        beta2=beta2,
        grad_clip=grad_clip,
        warmup_iters=warmup_iters,
        lr_decay_iters=lr_decay_iters,
        min_lr=min_lr,
        decay_lr=decay_lr
    )

    # save configs for future use
    with open(f"{outdir}/training_cfg.json","w") as fout:
        json.dump(training_cfg,fout,indent=4)
    with open(f"{outdir}/model_cfg.json","w") as fout:
        json.dump(model_cfg,fout,indent=4)

    # define model
    gpt_config = GPTConfig(**model_cfg)
    model = GPT(gpt_config)
    model.to(device)
    # compile model
    #model = torch.compile(model)

    sho_data = DampedSHODatasetXV(num_trajectories=num_train,seq_len=max_seq_length,min_seq_length=min_seq_length,vary_length=vary_length,
                                  masses=masses,dt=dt,k=k,beta=beta,pin_amplitude=pin_amplitude,
                                  min_amplitude=min_amplitude,k_context=k_context)
    val_data = DampedSHODatasetXV(num_trajectories=num_val,seq_len=max_seq_length,min_seq_length=min_seq_length,vary_length=vary_length,
                                  masses=masses,dt=dt,k=k,beta=beta,pin_amplitude=pin_amplitude,
                                  min_amplitude=min_amplitude,k_context=k_context)
    #sho_data = RandomSHODatasetVariableLength(masses,num_train,min_seq_length,max_seq_length,dt)
    #val_data = RandomSHODatasetVariableLength(masses,num_val,min_seq_length,max_seq_length,dt)
    #sho_data = RandomSHODatasetVariableLengthSmoothMass(masses,num_train,min_seq_length,max_seq_length,dt)
    #val_data = RandomSHODatasetVariableLengthSmoothMass(masses,num_val,min_seq_length,max_seq_length,dt)
    loader = DataLoader(sho_data,batch_size=bs,shuffle=True)
    val_loader = DataLoader(val_data,batch_size=bs_val,shuffle=True)

    optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)

    def get_lr(it):
        # 1) linear warmup for warmup_iters steps
        if it < warmup_iters:
            return learning_rate * it / warmup_iters
        # 2) if it > lr_decay_iters, return min learning rate
        if it > lr_decay_iters:
            return min_lr
        # 3) in between, use cosine decay down to min learning rate
        decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
        return min_lr + coeff * (learning_rate - min_lr)

    best_state = None
    best_loss = 1e9
    best_val_loss = 1e9
    best_train_loss = 1e9
    iter_count = 0
    loss_fn = nn.MSELoss()
    writer = SummaryWriter(log_dir=logdir)

    eps = range(num_epoch)
    with tqdm(total=len(eps), desc="Training") as pbar:
        for i in eps:
            model.train()
            train_losses = []
            local_best_loss = 1e9
            local_best_state = None
            for batch in loader:
                optimizer.zero_grad()
                if decay_lr:
                    lr = get_lr(iter_count)
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr

                input, target, context, mask = batch
                if torch.all(mask == -999):
                    mask = None
                if mask is not None:
                    inpt_mask, tgt_mask = mask[:,:-1,:].to(device), mask[:,1:,:].float().to(device)
                else:
                    inpt_mask, tgt_mask = None, torch.ones_like(input).to(device)
                preds = model(input.to(device),mask=inpt_mask)
                loss = loss_fn(preds*tgt_mask,target.to(device)*tgt_mask) # use target mask to zero out entries corresponding to sequence length padding
                loss.backward()
                if grad_clip != 0.0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
                train_losses.append(loss.item())

                if loss.item() < best_loss:
                    best_loss = loss.item()
                    best_state = model.state_dict()
                    torch.save(best_state,f"{outdir}/{model_name}_best.pt")
                if loss.item() < local_best_loss:
                    local_best_loss = loss.item()
                    local_best_state = model.state_dict()

                iter_count += 1
                pbar.set_description(f"Training, iter {iter_count}, {loss.item():.3e}")
                writer.add_scalar("IterLoss", loss.item(), iter_count)
            train_loss = np.mean(train_losses)
            if train_loss < best_train_loss:
                best_train_loss = train_loss
            torch.save(local_best_state,f"{outdir}/{model_name}_best_epoch_{i+1}.pt")

            model.eval()
            val_losses = []
            with torch.no_grad():
                for batch in val_loader:
                    input, target, context, mask = batch
                    if torch.all(mask == -999):
                        mask = None
                    if mask is not None:
                        inpt_mask, tgt_mask = mask[:,:-1,:].to(device), mask[:,1:,:].float().to(device)
                    else:
                        inpt_mask, tgt_mask = None, torch.ones_like(input).to(device)
                    preds = model(input.to(device),mask=inpt_mask)
                    loss = loss_fn(preds*tgt_mask,target.to(device)*tgt_mask)
                    val_losses.append(loss.item())
            val_loss = np.mean(val_losses)
            if val_loss < best_val_loss:
                best_val_loss = val_loss

            # updating tqdm
            pbar.set_postfix_str(f"Train {train_loss:.3e} (best {best_train_loss:.3e}), Val {val_loss:.3e} (best {best_val_loss:.3e})")
            pbar.update(1)

            # updating writer
            writer.add_scalars("Losses", {"Train":train_loss,"Val":val_loss}, i+1)
            
    writer.close()
    return model

if __name__ == "__main__":
    main()