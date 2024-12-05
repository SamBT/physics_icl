import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
import numpy as np

class SHODataset(Dataset):
    def __init__(self, masses, seq_len, dt, k=1.0):
        self.num_trajectories = len(masses)
        self.seq_len = seq_len
        self.masses = masses
        self.k = k
        self.dt = dt
        
        self.omegas = torch.sqrt(self.k/self.masses)
        self.rand_x0 = torch.rand(self.num_trajectories)*2 - 1
        self.rand_v0 = torch.sqrt((1.0 - self.rand_x0**2)*self.omegas**2)
        self.time_series, self.time = generate_harmonic_data(self.masses,
                                           self.rand_x0,
                                           self.rand_v0,
                                           self.dt,
                                           self.seq_len,
                                           k=self.k)
        self.time_series = self.time_series.unsqueeze(2) # make it have shape (num_trajectories, seq_len, 1)
        self.context = torch.cat([self.masses.unsqueeze(1),self.rand_x0.unsqueeze(1),self.rand_v0.unsqueeze(1)],dim=1)
        self.mask = None

    def __getitem__(self, idx):
        # Input sequence and next step
        input = self.time_series[idx,:-1]
        target = self.time_series[idx,1:]
        context = self.context[idx]
        mask = self.mask
        return input, target, context, mask
    
    def __len__(self):
        return len(self.time_series)

class RandomSHODataset(Dataset):
    def __init__(self, masses, num_trajectories, seq_len, dt, k=1.0):
        self.num_trajectories = num_trajectories
        self.seq_len = seq_len
        self.masses = masses
        self.k = k
        self.dt = dt
        
        self.rand_masses = self.masses[torch.randint(0,len(masses),(num_trajectories,))]
        self.omegas = torch.sqrt(self.k/self.rand_masses)
        self.rand_x0 = torch.rand(num_trajectories)*2 - 1
        self.rand_v0 = torch.sqrt((1.0 - self.rand_x0**2)*self.omegas**2)
        self.time_series, self.time = generate_harmonic_data(self.rand_masses,
                                           self.rand_x0,
                                           self.rand_v0,
                                           self.dt,
                                           self.seq_len,
                                           k=self.k)
        self.time_series = self.time_series.unsqueeze(2) # make it have shape (num_trajectories, seq_len, 1)
        self.context = torch.cat([self.rand_masses.unsqueeze(1),self.rand_x0.unsqueeze(1),self.rand_v0.unsqueeze(1)],dim=1)
        self.mask = None

    def __getitem__(self, idx):
        # Input sequence and next step
        input = self.time_series[idx,:-1]
        target = self.time_series[idx,1:]
        context = self.context[idx]
        mask = self.mask
        return input, target, context, mask
    
    def __len__(self):
        return len(self.time_series)
    
class RandomSHODatasetVariableLength(Dataset):
    def __init__(self, masses, num_trajectories, min_seq_len, max_seq_len, dt, k=1.0):
        self.num_trajectories = num_trajectories
        self.max_seq_length = max_seq_len
        self.min_seq_length = min_seq_len
        self.masses = masses
        self.k = k
        self.dt = dt
        
        self.rand_masses = self.masses[torch.randint(0,len(masses),(num_trajectories,))]
        self.omegas = torch.sqrt(self.k/self.rand_masses)
        self.rand_x0 = torch.rand(num_trajectories)*2 - 1
        self.rand_v0 = torch.sqrt((1.0 - self.rand_x0**2)*self.omegas**2)
        self.seq_lengths = torch.randint(self.min_seq_length,self.max_seq_length+1,(self.num_trajectories,))
        cols = torch.arange(self.max_seq_length).unsqueeze(0)
        self.mask = cols < self.seq_lengths.unsqueeze(1) # mask is True for entries that are part of the sequence
        #self.mask = self.mask.unsqueeze(2) # make it have shape (num_trajectories, max_seq_length, 1)

        self.time_series, self.time = generate_harmonic_data(self.rand_masses,
                                           self.rand_x0,
                                           self.rand_v0,
                                           self.dt,
                                           self.max_seq_length,
                                           k=self.k)
        
        self.time_series = self.time_series.unsqueeze(2) # make it have shape (num_trajectories, max_seq_len, 1)
        self.context = torch.cat([self.rand_masses.unsqueeze(1),self.rand_x0.unsqueeze(1),self.rand_v0.unsqueeze(1)],dim=1)

    def __getitem__(self, idx):
        # Input sequence and next step
        input = self.time_series[idx,:-1]
        target = self.time_series[idx,1:]
        context = self.context[idx]
        mask = self.mask[idx]
        return input, target, context, mask
    
    def __len__(self):
        return len(self.time_series)
    
class RandomSHODatasetVariableLengthSmoothMass(Dataset):
    def __init__(self, mass_ranges, num_trajectories, min_seq_len, max_seq_len, dt, k=1.0):
        self.num_trajectories = num_trajectories
        self.max_seq_length = max_seq_len
        self.min_seq_length = min_seq_len
        self.mass_ranges = mass_ranges # should be list of tuples
        self.k = k
        self.dt = dt

        self.rand_masses = random_intervals(self.mass_ranges,self.num_trajectories)
        self.omegas = torch.sqrt(self.k/self.rand_masses)
        self.rand_x0 = torch.rand(num_trajectories)*2 - 1
        self.rand_v0 = torch.sqrt((1.0 - self.rand_x0**2)*self.omegas**2)
        self.seq_lengths = torch.randint(self.min_seq_length,self.max_seq_length+1,(self.num_trajectories,))
        cols = torch.arange(self.max_seq_length).unsqueeze(0)
        self.mask = cols < self.seq_lengths.unsqueeze(1) # mask is True for entries that are part of the sequence
        #self.mask = self.mask.unsqueeze(2) # make it have shape (num_trajectories, max_seq_length, 1)

        self.time_series, self.time = generate_harmonic_data(self.rand_masses,
                                           self.rand_x0,
                                           self.rand_v0,
                                           self.dt,
                                           self.max_seq_length,
                                           k=self.k)
        
        self.time_series = self.time_series.unsqueeze(2) # make it have shape (num_trajectories, max_seq_len, 1)
        self.context = torch.cat([self.rand_masses.unsqueeze(1),self.rand_x0.unsqueeze(1),self.rand_v0.unsqueeze(1)],dim=1)

    def __getitem__(self, idx):
        # Input sequence and next step
        input = self.time_series[idx,:-1]
        target = self.time_series[idx,1:]
        context = self.context[idx]
        mask = self.mask[idx]
        return input, target, context, mask
    
    def __len__(self):
        return len(self.time_series)
    
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

# Generate synthetic data: simple harmonic motion
def generate_harmonic_data(m, x0, v0, dt, num_samples, k=1.0):
    bs = m.shape[0]
    assert m.shape[0] == x0.shape[0] == v0.shape[0]
    omega = torch.sqrt(k/m)
    c1 = x0
    c2 = v0/omega
    A = torch.sqrt(c1**2 + c2**2)
    phi = torch.arctan2(c2,c1)
    
    # reshape
    A = A.unsqueeze(1)
    omega = omega.unsqueeze(1)
    phi = phi.unsqueeze(1)
    time = torch.arange(0,num_samples*dt,dt)
    #time = torch.linspace(0, tmax, num_samples).repeat(bs).reshape(bs,-1)
    position = A * torch.cos(omega * time - phi)
    velocity = -A * omega * torch.sin(omega * time)
    return position, time

# Example data generation
#data = generate_harmonic_data(10000)
#dataset = TrajectoryDataset(data, seq_len=50)
#dataloader = DataLoader(dataset, batch_size=32, shuffle=True)