import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import math
import numpy as np
from tqdm import tqdm

from model import GPT, GPTConfig
#from physics_data import RandomSHODataset, RandomSHODatasetVariableLength, RandomSHODatasetVariableLengthSmoothMass
from PhysicsDatasets import SHODataset, SHODatasetXV, DampedSHODatasetXV, DampedSHODatasetV2
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import os
import json
import sys
import utils
import yaml

def main(config):
    has_cuda = torch.cuda.is_available()
    has_mps = torch.backends.mps.is_available()
    if has_cuda:
        device = 'cuda'
    elif has_mps:
        device = 'mps'
    else:
        device = 'cpu'
    device_type = device # for later use in torch.autocast

    # get the date/time for naming purposes
    now = datetime.now()
    formatted_date_time = now.strftime("%d%b%y_%H%M")
    model_name = f"random_DampedSHO_xvInput_noContext_v2_{formatted_date_time}"

    outdir = f"trainings/{config['model_name']}/{formatted_date_time}/"
    logdir = f"logs/{config['model_name']}/{formatted_date_time}/"

    train_params = config['training_params']
    dset_params = config['dataset_params']
    model_params = config['model_params']
    opt_params = config['opt_params']

    # train/val info
    num_train_iters = train_params['num_train_iters']
    save_every = train_params['save_every']
    val_every = train_params['val_every']
    num_val = train_params['num_val_seqs']
    bs = train_params['bs']
    bs_val = train_params['bs_val']
    num_val_iters = int(np.round(num_val/bs_val))

    # optimizer
    learning_rate = opt_params['lr']
    weight_decay = opt_params['weight_decay']
    beta1 = opt_params['beta1']
    beta2 = opt_params['beta2']
    grad_clip = opt_params['grad_clip']

    # learning rate schedule
    decay_lr = opt_params['decay_lr'] # whether to decay the learning rate
    warmup_iters = opt_params['warmup_iter_frac']*num_train_iters # how many steps to warm up for
    lr_decay_iters = opt_params['lr_decay_iter_frac']*num_train_iters # should be ~= max_iters per Chinchilla
    min_lr = opt_params['min_lr'] # minimum learning rate, should be ~= learning_rate/10 per Chinchilla

    # define model
    gpt_config = GPTConfig(**model_params)
    model = GPT(gpt_config)
    model.to(device)
    
    # compile model
    #model = torch.compile(model)

    sho_data = DampedSHODatasetV2(**dset_params)
    val_data = DampedSHODatasetV2(**dset_params)
    loader = iter(DataLoader(sho_data,batch_size=bs))
    val_loader = iter(DataLoader(val_data,batch_size=bs_val))

    optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)

    # make directories and save configs for future use
    if not os.path.isdir(outdir):
        os.makedirs(outdir)
    if not os.path.isdir(logdir):
        os.makedirs(logdir)

    with open(f"{outdir}/config.yaml","w") as fout:
        yaml.dump(config,fout)

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

    best_val_loss = 1e9
    best_train_loss = 1e9
    best_avg_loss = 1e9
    loss_fn = nn.MSELoss()
    writer = SummaryWriter(log_dir=logdir)

    train_losses = []
    coarse_train_losses = []
    with tqdm(total=num_train_iters, desc="Training") as pbar:
        for i in range(num_train_iters):
            optimizer.zero_grad()
            model.train()
            batch = next(loader)
            inpt, target, context, mask = batch
            
            if decay_lr:
                lr = get_lr(i)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
            
            if torch.all(mask == -999):
                mask = None
            if mask is not None:
                inpt_mask, tgt_mask = mask[:,:-1,:].float().to(device), mask[:,1:,:].float().to(device)
            else:
                inpt_mask, tgt_mask = None, torch.ones_like(inpt).to(device)
            
            preds = model(inpt.to(device),mask=inpt_mask)
            loss = loss_fn(preds*tgt_mask,target.to(device)*tgt_mask) # use target mask to zero out entries corresponding to sequence length padding
            loss.backward()
            if grad_clip != 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            train_losses.append(loss.item())

            if loss.item() < best_train_loss:
                best_train_loss = loss.item()
                torch.save(model.state_dict(),f"{outdir}/{model_name}_best.pt")
            
            pbar.set_description(f"Training, iter {i+1}, {loss.item():.3e}")
            writer.add_scalar("IterLoss", loss.item(), i)

            if i==0 or (i+1) % save_every == 0:
                torch.save(model.state_dict(),f"{outdir}/{model_name}_iter{i+1}.pt")
            
            if (i+1) % val_every == 0:
                avg_loss = np.mean(train_losses[-val_every:])
                if avg_loss < best_avg_loss:
                    best_avg_loss = avg_loss
                coarse_train_losses.append(avg_loss)
                
                model.eval()
                val_losses = []
                with torch.no_grad():
                    for j in range(num_val_iters):
                        batch = next(val_loader)
                        inpt, target, context, mask = batch
                        if torch.all(mask == -999):
                            mask = None
                        if mask is not None:
                            inpt_mask, tgt_mask = mask[:,:-1,:].to(device), mask[:,1:,:].float().to(device)
                        else:
                            inpt_mask, tgt_mask = None, torch.ones_like(inpt).to(device)
                        preds = model(inpt.to(device),mask=inpt_mask)
                        loss = loss_fn(preds*tgt_mask,target.to(device)*tgt_mask)
                        val_losses.append(loss.item())
                val_loss = np.mean(val_losses)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                pbar.set_postfix_str(f"Train {avg_loss:.3e} (best {avg_loss:.3e}), Val {val_loss:.3e} (best {best_val_loss:.3e})")
                writer.add_scalars("Losses", {"Train":avg_loss,"Val":val_loss}, (i+1) // val_every)
            
            pbar.update(1)
            
    writer.close()
    return model

if __name__ == "__main__":
    config = utils.load_config(sys.argv[1])
    main(config)