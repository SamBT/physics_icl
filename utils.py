import torch
import torch.nn as nn
import json
import numpy as np
import yaml
import os

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

def random_multiLinspace(intervals):
    i = intervals[np.random.choice(len(intervals))]
    assert type(i) == tuple and len(i) == 3
    return np.random.choice(np.linspace(i[0],i[1],i[2]))

def load_config(file):
    with open(file,"r") as fin:
        config = yaml.load(fin,Loader=yaml.FullLoader)
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
    
def forecast(model,input,n_steps,n_ctx,tokenized=False):
    inpt_sel = input[:,:n_ctx]
    with torch.no_grad():
        for i in range(n_steps):
            pred = model(inpt_sel)
            if tokenized:
                pred = torch.argmax(pred,dim=-1)
            inpt_sel = torch.cat([inpt_sel,pred[:,-1:]],dim=1)
    return inpt_sel

def get_freqs(series,dt):
    fft_result = np.fft.fft(series)
    sampling_rate = 1.0/dt
    frequencies = np.fft.fftfreq(len(series), d=1/sampling_rate)
    magnitude = np.abs(fft_result)
    positive_freqs = frequencies[:len(frequencies)//2]  # Only positive frequencies
    positive_magnitude = magnitude[:len(magnitude)//2]  # Corresponding magnitudes
    dominant_freq = positive_freqs[np.argmax(positive_magnitude)]*2*np.pi

    print(f"Dominant w0: {dominant_freq}")
    
    return dominant_freq