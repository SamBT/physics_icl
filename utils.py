import torch
import torch.nn as nn
from model import GPT, GPTConfig
import json
import numpy as np
from types import SimpleNamespace
import yaml
import os

def load_model(name,date,epoch=None):
    src = f"trainings/{name}_{date}/"
    with open(f"{src}/model_cfg.json","r") as fin:
        model_cfg = json.load(fin)
    with open(f"{src}/training_cfg.json","r") as fin:
        train_cfg = json.load(fin)

    dt = train_cfg['dt']
    k = train_cfg['k']
    beta = train_cfg['beta']
    pin_amplitude = train_cfg['pin_amplitude']
    min_amplitude = train_cfg['min_amplitude']
    k_context = train_cfg['k_context']

    # define model
    gpt_config = GPTConfig(**model_cfg)
    model = GPT(gpt_config)
    if epoch is None:
        model_name = f"{name}_{date}_best.pt"
    else:
        model_name = f"{name}_{date}_best_epoch_{epoch}.pt"
    model.load_state_dict(torch.load(f"{src}/{model_name}",map_location='cpu'))
    model.eval()

    kwargs = dict(
        dt=dt,
        k=k,
        beta=beta,
        pin_amplitude=pin_amplitude,
        min_amplitude=min_amplitude,
        k_context=k_context
    )

    return model, kwargs

def get_model_v2(config):
    gpt_config = GPTConfig(**config['model_params'])
    model = GPT(gpt_config)
    return model

def load_model_v2(config,tgt_dir,ckpt='best'):
    models = [m for m in os.listdir(tgt_dir) if ckpt in m and '.pt' in m]
    assert len(models) == 1
    model = get_model_v2(config)
    model.load_state_dict(torch.load(f"{tgt_dir}/{models[0]}",map_location='cpu'))
    model.eval()
    return model

def random_intervals(intervals,num_samples):
    output = []
    num_per = num_samples//len(intervals)
    rem = num_samples % len(intervals)
    nums = [num_per]*len(intervals)
    for i in range(rem):
        nums[i] += 1
    for i,r in enumerate(intervals):
        assert r[1] > r[0]
        rint = np.rand(nums[i])
        rint = rint * (r[1]-r[0]) + r[0]
        output.append(rint)
    output = np.cat(output,dim=0)
    output = output[np.randperm(output.shape[0])]
    return output

def random_multiInterval(intervals):
    i = intervals[np.random.choice(len(intervals))]
    assert type(i) == tuple and len(i) == 2
    return i[0] + np.random.rand()*(i[1] - i[0])

def load_config(file):
    with open(file,"r") as fin:
        config = yaml.load(fin,Loader=yaml.FullLoader)
    config = parse_config(config)
    return config

def tuplify(l):
    if type(l[0]) == list:
        assert np.all([len(li)==2 for li in l])
        return [tuple(li) for li in l]
    else:
        assert len(l) == 2
        return tuple(l)

def parse_config(config):    
    if config['parsing_params']['m_tuple']:
        config['dataset_params']['m'] = tuplify(config['dataset_params']['m'])
    if config['parsing_params']['k_tuple']:
        config['dataset_params']['k'] = tuplify(config['dataset_params']['k'])
    if config['parsing_params']['beta_tuple']:
        config['dataset_params']['beta'] = tuplify(config['dataset_params']['beta'])
    
    return config

class RealNumberTokenizer:
    def __init__(self, num_bins, range_limit):
        """
        Tokenizer for real numbers using configurable binning.

        Args:
            num_bins (int): Number of bins to divide the range into.
            range_limit (float): Symmetric range limit [-range_limit, range_limit].
        """
        self.num_bins = num_bins
        self.range_limit = range_limit
        self.bin_edges = torch.linspace(-range_limit, range_limit, num_bins + 1)  # Bin edges including overflow/underflow

    def tokenize(self, values):
        """
        Tokenizes real numbers into bin indices.

        Args:
            values (torch.Tensor): Tensor of real numbers to tokenize.

        Returns:
            torch.Tensor: Tensor of bin indices (0 to num_bins-1).
        """
        # Assign each value to a bin
        bin_indices = torch.bucketize(values, self.bin_edges) - 1  # `bucketize` returns indices relative to edges
        # Clamp indices to valid bin range [0, num_bins-1]
        bin_indices = torch.clamp(bin_indices, 0, self.num_bins - 1)
        return bin_indices

    def detokenize(self, bin_indices):
        """
        Converts bin indices back to representative real numbers.

        Args:
            bin_indices (torch.Tensor): Tensor of bin indices.

        Returns:
            torch.Tensor: Tensor of representative real numbers (bin centers).
        """
        bin_indices = torch.clamp(bin_indices, 0, self.num_bins - 1)
        bin_centers = (self.bin_edges[:-1] + self.bin_edges[1:]) / 2  # Compute bin centers
        return bin_centers[bin_indices]