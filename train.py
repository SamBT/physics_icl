import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import math
import numpy as np
from tqdm import tqdm

from model import GPT, GPTConfig
from PhysicsDatasets import SHODataset, SHODatasetXV, DampedSHODatasetXV, DampedSHODatasetV2
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import os
import json
import sys
import utils
import yaml
from concurrent.futures import ThreadPoolExecutor
from argparse import ArgumentParser
import copy

def main(model_config, dataset_config, training_config, opt_config):
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
    # get names from configs
    model_name = model_config['name']
    dataset_name = dataset_config['name']
    training_name = training_config['name']
    opt_name = opt_config['name']
    # whether to tokenize or not
    tokenized = model_config['GPT']['tokenized']

    # make directories
    top_dir = "mse" if not tokenized else "tok"
    dir_path = f"{top_dir}/{model_name}/train_{dataset_name}/{training_name}_{opt_name}/"

    outdir = f"trainings/{dir_path}/"
    logdir = f"logs/{dir_path}/"

    # train/val info
    num_train_iters = training_config['num_train_iters']
    save_every = training_config['save_every']
    val_every = training_config['val_every']
    num_val = training_config['num_val_seqs']
    bs = training_config['bs']
    bs_val = training_config['bs_val']
    seq_len = training_config['seq_len']
    num_val_iters = int(np.round(num_val/bs_val))

    # optimizer
    learning_rate = opt_config['lr']
    weight_decay = opt_config['weight_decay']
    beta1 = opt_config['beta1']
    beta2 = opt_config['beta2']
    grad_clip = opt_config['grad_clip']

    # learning rate schedule
    decay_lr = opt_config['decay_lr'] # whether to decay the learning rate
    warmup_iters = opt_config['warmup_iter_frac']*num_train_iters # how many steps to warm up for
    lr_decay_iters = opt_config['lr_decay_iter_frac']*num_train_iters # should be ~= max_iters per Chinchilla
    min_lr = opt_config['min_lr'] # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
    warmup = opt_config['warmup'] # whether to do linear warmup
    cos_anneal = opt_config['cos_anneal'] # whether to do cosine annealing

    # define model
    gpt_config = GPTConfig(**model_config['GPT'])
    model = GPT(gpt_config)
    model.to(device)

    if tokenized:
        tokenizer = utils.RealNumberTokenizer(model_config['GPT']['vocab_size'],model_config['GPT']['range_limit_tok'])
    
    # compile model
    #model = torch.compile(model)

    dataset_config['seq_len'] = seq_len
    sho_data = DampedSHODatasetV2(**dataset_config)
    val_data = DampedSHODatasetV2(**dataset_config)
    loader = iter(DataLoader(sho_data,batch_size=bs))
    val_loader = iter(DataLoader(val_data,batch_size=bs_val))

    # make directories and save configs for future use
    if not os.path.isdir(outdir):
        os.makedirs(outdir)
    if not os.path.isdir(logdir):
        os.makedirs(logdir)

    # define parameter space for testing
    test_w0s = np.arange(0.1,7.1,0.1)
    test_betas = np.arange(0,7.5,0.5)
    np.save(f"{outdir}/test_w0s.npy",test_w0s)
    np.save(f"{outdir}/test_betas.npy",test_betas)

    optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)

    full_config = {"model":model_config,
                   "dataset":dataset_config,
                   "training":training_config,
                   "opt":opt_config}
    with open(f"{outdir}/config.yaml","w") as fout:
        yaml.dump(full_config,fout)

    def get_lr(it):
        # warmup, if applicable
        if it < warmup_iters and warmup:
            return learning_rate * it / warmup_iters
        # cosine-annealing, if applicable
        if cos_anneal:
            # if it > lr_decay_iters, return min learning rate
            if it > lr_decay_iters:
                return min_lr
            # in between, use cosine decay down to min learning rate
            lower_bound = warmup_iters if warmup else 0
            decay_ratio = (it - lower_bound) / (lr_decay_iters - lower_bound)
            assert 0 <= decay_ratio <= 1
            coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
            return min_lr + coeff * (learning_rate - min_lr)
        # otherwise, return the fixed learning rate
        return learning_rate

    best_val_loss = 1e9
    best_train_loss = 1e9
    best_avg_loss = 1e9
    if tokenized:
        loss_fn = F.cross_entropy
    else:
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
            if tokenized:
                inpt = tokenizer.tokenize(inpt).squeeze(-1)
                target = tokenizer.tokenize(target).squeeze(-1)
                mask = mask.squeeze(-1)
            
            if decay_lr:
                lr = get_lr(i)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
            
            if torch.all(mask == -999):
                mask = None
            if mask is not None:
                inpt_mask, tgt_mask = mask[:,:-1].float().to(device), mask[:,1:].float().to(device)
            else:
                inpt_mask, tgt_mask = None, torch.ones_like(inpt).to(device)
            
            preds = model(inpt.to(device),mask=inpt_mask)
            if tokenized:
                loss = loss_fn(preds.view(-1,preds.size(-1)),target.view(-1).to(device),reduction='none')
                loss = (tgt_mask.view(-1)*loss).mean()
            else:
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
                    # evaluate on val set
                    for j in range(num_val_iters):
                        batch = next(val_loader)
                        inpt, target, context, mask = batch
                        if tokenized:
                            inpt = tokenizer.tokenize(inpt).squeeze(-1)
                            target = tokenizer.tokenize(target).squeeze(-1)
                            mask = mask.squeeze(-1)
                        if torch.all(mask == -999):
                            mask = None
                        if mask is not None:
                            inpt_mask, tgt_mask = mask[:,:-1].to(device), mask[:,1:].float().to(device)
                        else:
                            inpt_mask, tgt_mask = None, torch.ones_like(inpt).to(device)
                        preds = model(inpt.to(device),mask=inpt_mask)
                        if tokenized:
                            loss = loss_fn(preds.view(-1,preds.size(-1)),target.view(-1).to(device),reduction='none')
                            loss = (tgt_mask.view(-1)*loss).mean()
                        else:
                            loss = loss_fn(preds*tgt_mask,target.to(device)*tgt_mask)
                        val_losses.append(loss.item())
                    
                val_loss = np.mean(val_losses)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                pbar.set_postfix_str(f"Train {avg_loss:.3e} (best {avg_loss:.3e}), Val {val_loss:.3e} (best {best_val_loss:.3e})")
                writer.add_scalars("Losses", {"Train":avg_loss,"Val":val_loss}, (i+1) // val_every)
                
                # test evaluations with full parameter range
                with torch.no_grad():
                    all_inpts = []
                    all_tgts = []
                    n_per = 100
                    for w0 in test_w0s:
                        for beta in test_betas:
                            cfg = copy.deepcopy(dataset_config)
                            cfg['w0'] = w0
                            cfg['beta'] = beta
                            cfg['vary_length'] = False
                            dset = DampedSHODatasetV2(**cfg)
                            ipt,tgt = dset.sample(n_per)
                            all_inpts.append(ipt)
                            all_tgts.append(tgt)
                    bs_test = 1024
                    all_inpts = torch.split(torch.cat(all_inpts,dim=0),bs_test)
                    all_tgts = torch.split(torch.cat(all_tgts,dim=0),bs_test)
                    all_losses = []
                    for inpt, tgt in zip(all_inpts,all_tgts):
                        if tokenized:
                            inpt = tokenizer.tokenize(inpt).squeeze(-1)
                            tgt = tokenizer.tokenize(tgt).squeeze(-1)
                        preds = model(inpt.to(device))
                        if tokenized:
                            loss = F.cross_entropy(preds.view(-1,preds.size(-1)),tgt.view(-1).to(device),reduction='none').view(-1,tgt.size(-1)).cpu().numpy()
                        else:
                            loss = F.mse_loss(preds,tgt.to(device),reduction='none').sum(dim=-1).cpu().numpy()
                        all_losses.append(loss)
                    all_losses = np.concatenate(all_losses,axis=0)
                    all_losses = all_losses.reshape(len(test_w0s),len(test_betas),n_per,all_losses.shape[-1]).mean(axis=-2)
                    np.save(f"{outdir}/test_losses_iter{i+1}.npy",all_losses)
                    del all_losses, all_inpts, all_tgts
            
            pbar.update(1)
            
    writer.close()
    return model

def run(model_config,dataset_config,training_config,opt_config):
    model_config = utils.load_config(model_config)
    dataset_config = utils.load_config(dataset_config)
    training_config = utils.load_config(training_config)
    opt_config = utils.load_config(opt_config)
    
    main(model_config, dataset_config, training_config, opt_config)

if __name__ == "__main__":
    parser = ArgumentParser(description="Configuration for training")

    # Add arguments
    parser.add_argument("-m","--model", type=str, help="model config yaml",required=True)
    parser.add_argument("-d","--dataset", type=str, help="dataset config yaml",required=True)
    parser.add_argument("-t","--training", type=str, help="training config yaml",required=True)
    parser.add_argument("-o","--opt", type=str, help="optimizer config yaml",required=True)

    # Parse arguments
    args = parser.parse_args()
    
    run(args.model,args.dataset,args.training,args.opt)
    
    """if len(sys.argv) > 2:
        yamls = sys.argv[1:]
        with ThreadPoolExecutor(max_workers=len(yamls)) as executor:
            executor.map(run,yamls)
    else:
        run(sys.argv[1])"""