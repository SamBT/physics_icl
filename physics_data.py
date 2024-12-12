import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
import numpy as np

# Generate synthetic data: simple harmonic motion
def generate_harmonic_data(m, x0, v0, dt, num_samples, k=1.0):
    bs = m.shape[0]
    assert m.shape[0] == x0.shape[0] == v0.shape[0]
    omega = torch.sqrt(k/m)
    c1 = x0
    c2 = v0/omega
    # reshape & make time series
    c1 = c1.unsqueeze(1)
    c2 = c2.unsqueeze(1)
    omega = omega.unsqueeze(1)
    time = torch.arange(0,num_samples*dt,dt)
    position = c1 * torch.cos(omega*time) + c2 * torch.sin(omega*time)
    velocity = -omega*c1*torch.sin(omega*time) + omega*c2*torch.cos(omega*time)
    return position, velocity, time

def random_intervals(intervals,num_samples):
    output = []
    num_per = num_samples//len(intervals)
    rem = num_samples % len(intervals)
    nums = [num_per]*len(intervals)
    for i in range(rem):
        nums[i] += 1
    for i,r in enumerate(intervals):
        assert r[1] > r[0]
        rint = torch.rand(nums[i])
        rint = rint * (r[1]-r[0]) + r[0]
        output.append(rint)
    output = torch.cat(output,dim=0)
    output = output[torch.randperm(output.shape[0])]
    return output