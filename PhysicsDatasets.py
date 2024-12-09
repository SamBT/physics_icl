import physics_data as pdata
import torch
import numpy as np
from torch.utils.data import Dataset
import sys
from collections.abc import Iterable

class SHODataset(Dataset):
    def __init__(self, dt=0.1, seq_len=50, num_trajectories=None, masses=None, x0=None, v0=None, k=10.0, pin_amplitude=None, min_amplitude=0.1):
        if num_trajectories is not None:
            self.num_trajectories = num_trajectories
        else:
            if masses is not None and isinstance(masses,Iterable):
                if type(masses[0]) != tuple:
                    self.num_trajectories = len(masses)
            elif x0 is not None or v0 is not None:
                self.num_trajectories = len(x0) if x0 is not None else len(v0)
            else:
                print("Error: got no compatible input to specify num_trajectories!")
                sys.exit()
        self.dt = dt
        self.k = k
        self.seq_len = seq_len
        self.tmax = self.dt * self.seq_len
        
        # generating random parameters
        mmin, mmax = 0.1, 10
        xmin, xmax = -1, 1
        vmin, vmax = -1, 1
        rand_masses = torch.rand(self.num_trajectories)*(mmax-mmin) + mmin
        rand_x0 = torch.rand(self.num_trajectories)*(xmax-xmin) + xmin
        rand_v0 = torch.rand(self.num_trajectories)*(vmax-vmin) + vmin

        # setting parameters based on inputs
        # masses
        if masses is not None:
            if isinstance(masses,Iterable):
                if type(masses[0]) == tuple:
                    self.masses = pdata.random_intervals(masses,self.num_trajectories)
                else:
                    self.masses = masses
            else:
                self.masses = masses*torch.ones(self.num_trajectories)
        else:
            self.masses = rand_masses
        # initial positions
        if x0 is None:
            self.x0 = rand_x0
        else:
            if type(x0) != torch.Tensor:
                self.x0 = x0*torch.ones(self.num_trajectories)
            else:
                self.x0 = x0
        # initial velocities
        if v0 is None:
            self.v0 = rand_v0
        else:
            if type(v0) != torch.Tensor:
                self.v0 = v0*torch.ones(self.num_trajectories)
            else:
                self.v0 = v0
        # compute frequency & change v0 to fix amplitude if requested
        self.omegas = torch.sqrt(self.k/self.masses)
        if pin_amplitude is not None:
            assert pin_amplitude >= min_amplitude
            """if v0 is not None:
                sign = torch.sign(self.v0)
            else:
                sign = torch.randint(0,2,(len(self.v0),))*2 - 1"""
            sign = torch.sign(self.v0)
            self.v0 = sign * torch.sqrt(self.omegas**2*(pin_amplitude**2 - self.x0**2))
        self.A = torch.sqrt(self.x0**2 + (self.v0/self.omegas)**2)
        if torch.any(self.A < min_amplitude):
            self.A = torch.clamp(self.A,min=min_amplitude)
            self.v0 = torch.sign(self.v0)*torch.sqrt(self.omegas**2*(self.A**2 - self.x0**2))

        assert len(self.masses) == len(self.x0) == len(self.v0)
        
        # make time series data
        self.xt, self.vt, self.time = pdata.generate_harmonic_data(self.masses,
                                            self.x0,
                                            self.v0,
                                            self.dt,
                                            self.seq_len,
                                            k=self.k)
        self.norm_masses = (self.masses-5)/5 # assume mass varies in the range (0,10)
        self.xt = self.xt.unsqueeze(2) # make it have shape (num_trajectories, seq_len, 1)
        self.vt = self.vt.unsqueeze(2)
        self.context = torch.cat([self.norm_masses.unsqueeze(1),self.x0.unsqueeze(1),self.v0.unsqueeze(1)],dim=1)
        self.mask = -999

    def __getitem__(self, idx):
        # Input sequence and next step
        input = self.xt[idx,:-1]
        target = self.xt[idx,1:]
        context = self.context[idx]
        mask = self.mask
        return input, target, context, mask
    
    def __len__(self):
        return len(self.xt)
    
class SHODatasetXV(SHODataset):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        self.xv = torch.cat([self.xt,self.vt],dim=-1)
        self.context = self.norm_masses.unsqueeze(1)

    def __getitem__(self, idx):
        # Input sequence and next step

        input = self.xv[idx,:-1]
        target = self.xv[idx,1:]
        context = self.context[idx]
        mask = self.mask
        return input, target, context, mask