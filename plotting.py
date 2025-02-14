import numpy as np
import torch
import torch.nn.functional as F
import utils
from datetime import datetime
import os
import re
from PhysicsDatasets import DampedSHODatasetXV


def get_latest_model_dir(model_name,base_dir='trainings'):
    runs = [s for s in os.listdir(f"{base_dir}/{model_name}/")]
    runs_time = [datetime.strptime(s,"%d%b%y_%H%M") for s in runs]
    imax = max(enumerate(runs_time), key=lambda x: x[1])[0]
    run = runs[imax]
    model_dir = f"{base_dir}/{model_name}/{run}/"
    return model_dir


def load_all_models(model_name,base_dir='trainings',quiet=True,version=None):
    if version is None:
        model_dir = get_latest_model_dir(model_name,base_dir=base_dir)
    else:
        model_dir = f"{base_dir}/{model_name}/{version}/"

    config = utils.load_config(f"{model_dir}/config.yaml")
    best_model = utils.load_model_v2(config,model_dir,ckpt='best',quiet=quiet)
    all_models = [f for f in os.listdir(model_dir) if '.pt' in f and 'iter' in f]
    iter_models = {}
    iter_ckpts = []
    for m in all_models:
        n_iter = int(re.search("iter(\\d+)",m).group(1))
        if n_iter == 1:
            continue
        iter_models[n_iter] = utils.load_model_v2(config,model_dir,name=m,quiet=quiet)
        iter_ckpts.append(n_iter)
    iter_ckpts = sorted(iter_ckpts)    

    return best_model, iter_models, iter_ckpts, config, model_dir

def make_dataset(masses,ks,betas,config,seq_len,num_trajectories_per,xv=False,tokenize=True):
    dt = config['dataset_params']['dt']

    all_m = -1*np.ones((num_trajectories_per,len(masses),len(ks),len(betas)))
    all_k = -1*np.ones((num_trajectories_per,len(masses),len(ks),len(betas)))
    all_beta = -1*np.ones((num_trajectories_per,len(masses),len(ks),len(betas)))
    for im,m in enumerate(masses):
        for ik,k in enumerate(ks):
            for ib,b in enumerate(betas):
                all_m[:,im,ik,ib] = m
                all_k[:,im,ik,ib] = k
                all_beta[:,im,ik,ib] = b
    shape = all_m.shape

    dset = DampedSHODatasetXV(masses=torch.tensor(all_m.flatten()),
                            k=torch.tensor(all_k.flatten()),
                            beta=torch.tensor(all_beta.flatten()),
                            seq_len=seq_len,dt=dt)
    del all_m, all_k, all_beta
    data = dset.xv if xv else dset.xt

    if tokenize:
        tokenizer = utils.RealNumberTokenizer(config['model_params']['vocab_size'], config['training_params']['range_limit_tok'])
        data = tokenizer.tokenize(data.squeeze(-1))

    return data, shape

def get_samples(dataset,N):
    output = []
    iter_d = iter(dataset)
    for i in range(N):
        output.append(next(iter_d))
    inpt = torch.cat([o[0].unsqueeze(0) for o in output],dim=0)
    target = torch.cat([o[1].unsqueeze(0) for o in output],dim=0)
    return inpt, target

def evaluate(model,data,shape,device,metric="cross_entropy",bs=10000):
    losses = []
    model = model.to(device)
    with torch.no_grad():
        for batch in torch.split(data,bs):
            pred = model(batch.to(device,dtype=torch.float32)).cpu()[:,:-1]
            tgt = batch[:,1:]

            if metric == "mse":
                loss = ((pred - tgt)**2).numpy().sum(axis=-1)
            elif metric == "cross_entropy":
                loss = F.cross_entropy(pred.reshape(-1,pred.size(-1)),tgt.reshape(-1),reduction='none').view(-1,tgt.size(-1)).numpy()

            losses.append(loss)

    loss = np.concatenate(losses,axis=0)
    loss = loss.reshape(*shape,loss.shape[-1]).mean(axis=0).transpose((3,0,1,2))
    return loss

def load_evaluations(model,base_dir="trainings"):
    data = {}
    model_dir = get_latest_model_dir(model,base_dir=base_dir)
    results = [f for f in os.listdir(model_dir) if "iter" in f and ".npy" in f]
    n_iters = [int(re.search("iter(\\d+)",m).group(1)) for m in results]
    for n_iter in n_iters:
        data[n_iter] = np.load(f"{model_dir}/eval_iter{n_iter}.npy")

    masses = np.load(f"{model_dir}/eval_masses.npy")
    ks = np.load(f"{model_dir}/eval_ks.npy")
    betas = np.load(f"{model_dir}/eval_betas.npy")
    
    return data, sorted(n_iters), masses, ks, betas



