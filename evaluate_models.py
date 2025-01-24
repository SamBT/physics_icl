from PhysicsDatasets import SHODatasetXV, DampedSHODatasetXV, DampedSHODatasetV2
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import json
from matplotlib.colors import LogNorm, Normalize
import utils
import os
from datetime import datetime
import yaml
import re
from tqdm import tqdm
import plotting as ptools
import sys

device = 'cuda' if torch.cuda.is_available() else 'cpu'

models = [m[:-1] if m[-1] == "/" else m for m in sys.argv[1:] ]
models = [f.split("/")[-1] for f in models]

masses = np.concatenate([np.array([0.1,0.5,0.75]),np.arange(1,13)])
ks = np.arange(8,23)
betas = np.arange(7)
seq_len = 100
num_trajectories_per = 100

for model in models:
    print(model)
    metric = "cross_entropy" if "tokenized" in model else "mse"
    best_model, iter_models, iter_ckpts, config, model_dir = ptools.load_all_models(model)

    data, shape = ptools.make_dataset(masses,ks,betas,config,seq_len,num_trajectories_per,xv=True,tokenize=False)

    np.save(f"{model_dir}/eval_masses.npy",masses)
    np.save(f"{model_dir}/eval_ks.npy",ks)
    np.save(f"{model_dir}/eval_betas.npy",betas)

    bs = 50_000

    loss = ptools.evaluate(best_model,data,shape,device,metric=metric,bs=bs)
    np.save(f"{model_dir}/eval_best.npy",loss)
    for n_iter in tqdm(iter_ckpts):
        loss = ptools.evaluate(iter_models[n_iter],data,shape,device,metric=metric,bs=bs)
        np.save(f"{model_dir}/eval_iter{n_iter}.npy",loss)

    del data
    del best_model
    del iter_models
    torch.cuda.empty_cache()