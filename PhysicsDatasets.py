import physics_data as pdata
import torch
import numpy as np
from torch.utils.data import Dataset, IterableDataset
import sys
from collections.abc import Iterable
import physics_data_v2 as pdata2
import utils

class SHODataset(Dataset):
    def __init__(self, dt=0.1, seq_len=50, num_trajectories=None, masses=None, x0=None, v0=None, k=10.0, pin_amplitude=None, min_amplitude=0.1, k_context=False):
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
        self.k_context = k_context
        
        # generating random parameters
        mmin, mmax = 0.1, 10
        xmin, xmax = -1, 1
        vmin, vmax = -1, 1
        kmin, kmax = 5, 15
        rand_masses = torch.rand(self.num_trajectories)*(mmax-mmin) + mmin
        rand_x0 = torch.rand(self.num_trajectories)*(xmax-xmin) + xmin
        rand_v0 = torch.rand(self.num_trajectories)*(vmax-vmin) + vmin
        rand_k = torch.rand(self.num_trajectories)*(kmax-kmin) + kmin

        # setting parameters based on inputs
        # random k if desired
        if self.k is None:
            self.k = rand_k
        else:
            self.k = self.k * torch.ones(self.num_trajectories)
        # masses
        if masses is not None:
            if isinstance(masses,Iterable):
                if type(masses[0]) == tuple:
                    self.masses = utils.random_intervals(masses,self.num_trajectories)
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
            sign = torch.sign(self.v0)
            self.v0 = sign * torch.sqrt(self.omegas**2*(pin_amplitude**2 - self.x0**2))
        self.A = torch.sqrt(self.x0**2 + (self.v0/self.omegas)**2)
        if torch.any(self.A < min_amplitude):
            self.A = torch.clamp(self.A,min=min_amplitude)
            self.v0 = torch.sign(self.v0)*torch.sqrt(self.omegas**2*(self.A**2 - self.x0**2))

        assert len(self.masses) == len(self.x0) == len(self.v0)
        
        # make time series data
        self.xt, self.vt, self.time = pdata.generate_harmonic_data(self.masses,
                                            self.k,
                                            self.x0,
                                            self.v0,
                                            self.dt,
                                            self.seq_len)
        self.norm_masses = (self.masses-5)/5 # assume mass varies in the range (0,10)
        self.norm_k = (self.k-10)/5 # assume k varies in the range (5,15)
        self.xt = self.xt.unsqueeze(2) # make it have shape (num_trajectories, seq_len, 1)
        self.vt = self.vt.unsqueeze(2)
        self.context = torch.cat([self.norm_masses.unsqueeze(1),self.x0.unsqueeze(1),self.v0.unsqueeze(1)],dim=1)
        if self.k_context:
            self.context = torch.cat([self.context,self.norm_k.unsqueeze(1)],dim=1)
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
        self.context = self.norm_masses.unsqueeze(1) # don't want (x0,v0) in context token
        if self.k_context:
            self.context = torch.cat([self.context,self.norm_k.unsqueeze(1)],dim=1)


    def __getitem__(self, idx):
        # Input sequence and next step

        input = self.xv[idx,:-1]
        target = self.xv[idx,1:]
        context = self.context[idx]
        mask = self.mask
        return input, target, context, mask
    
class DampedSHODataset(Dataset):
    def __init__(self, dt=0.1, seq_len=50, min_seq_length=20, num_trajectories=None, masses=None, x0=None, v0=None, k=None, beta=None, pin_amplitude=None, min_amplitude=None, k_context=False, vary_length=False):
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
        self.beta = beta
        self.seq_len = seq_len
        self.min_seq_length = min_seq_length
        self.tmax = self.dt * self.seq_len
        self.k_context = k_context
        self.vary_length = vary_length
        
        # generating random parameters
        mmin, mmax = 1, 10
        xmin, xmax = -1, 1
        vmin, vmax = -1, 1
        kmin, kmax = 10, 20
        betamin, betamax = 0, 3.7 # to make roughly half samples be overdamped, half underdamped when k and m are sampled as above
        rand_masses = torch.rand(self.num_trajectories)*(mmax-mmin) + mmin
        rand_x0 = torch.rand(self.num_trajectories)*(xmax-xmin) + xmin
        rand_v0 = torch.rand(self.num_trajectories)*(vmax-vmin) + vmin
        rand_k = torch.rand(self.num_trajectories)*(kmax-kmin) + kmin
        rand_beta = torch.rand(self.num_trajectories)*(betamax-betamin) + betamin

        # setting parameters based on inputs
        # random k if desired
        if self.k is None:
            self.k = rand_k
        elif isinstance(self.k,Iterable):
            if type(k[0]) == tuple:
                self.k = utils.random_intervals(self.k,self.num_trajectories)
            else:
                self.k = self.k
        else:
            self.k = self.k * torch.ones(self.num_trajectories)
        # random damping if desired
        if self.beta is None:
            self.beta = rand_beta
        elif isinstance(self.beta,Iterable):
            if type(self.beta[0]) == tuple:
                self.beta = utils.random_intervals(self.beta,self.num_trajectories)
            else:
                self.beta = self.beta
        else:
            self.beta = self.beta * torch.ones(self.num_trajectories)
        # masses
        if masses is not None:
            if isinstance(masses,Iterable):
                if type(masses[0]) == tuple:
                    self.masses = utils.random_intervals(masses,self.num_trajectories)
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
            if min_amplitude is not None:
                assert pin_amplitude >= min_amplitude
            sign = torch.sign(self.v0)
            self.v0 = sign * torch.sqrt(self.omegas**2*(pin_amplitude**2 - self.x0**2))
        self.A = torch.sqrt(self.x0**2 + (self.v0/self.omegas)**2)
        if min_amplitude is not None:
            if torch.any(self.A < min_amplitude):
                self.A = torch.clamp(self.A,min=min_amplitude)
                self.v0 = torch.sign(self.v0)*torch.sqrt(self.omegas**2*(self.A**2 - self.x0**2))

        assert len(self.masses) == len(self.x0) == len(self.v0)
        
        # make time series data
        self.xt, self.vt, self.time = pdata.generate_damped_harmonic_data(self.masses,
                                            self.k,
                                            self.beta,
                                            self.x0,
                                            self.v0,
                                            self.dt,
                                            self.seq_len)
        # record the damping status of each time series
        self.mask_status = self.compute_damping_status()
        self.norm_masses = (self.masses-5)/5 # assume mass varies in the range (0,10)
        self.norm_k = (self.k-10)/5 # assume k varies in the range (5,15)
        self.xt = self.xt.unsqueeze(2) # make it have shape (num_trajectories, seq_len, 1)
        self.vt = self.vt.unsqueeze(2)
        self.context = torch.cat([self.norm_masses.unsqueeze(1),self.x0.unsqueeze(1),self.v0.unsqueeze(1)],dim=1)
        if self.k_context:
            self.context = torch.cat([self.context,self.norm_k.unsqueeze(1)],dim=1)

        # set what data we want to use as inputs
        self.inputs = self.xt
        
        # create mask if we want variable-length sequences (e.g. training)
        if self.vary_length:
            lengths = torch.randint(self.min_seq_length,self.seq_len+1,(self.num_trajectories,)).reshape(-1,1)
            self.mask = torch.arange(self.seq_len).reshape(1,-1).tile(self.num_trajectories,1)
            self.mask = (self.mask < lengths).float().unsqueeze(-1)
        else:
            self.mask = -999*torch.ones(self.num_trajectories)

    def __getitem__(self, idx):
        # Input sequence and next step
        input = self.inputs[idx,:-1]
        target = self.inputs[idx,1:]
        context = self.context[idx]
        mask = self.mask[idx]
        return input, target, context, mask
    
    def __len__(self):
        return len(self.xt)
    
    def compute_damping_status(self):
        w0 = torch.sqrt(self.k/self.masses)
        mask_undamp = 1*(self.beta==0).float()
        mask_underdamp = 2*((self.beta<w0)&(self.beta>0)).float()
        mask_critdamp = 3*(self.beta==w0).float()
        mask_overdamp = 4*(self.beta>w0).float()
        return mask_undamp+mask_underdamp+mask_critdamp+mask_overdamp
    
class DampedSHODatasetXV(DampedSHODataset):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        self.xv = torch.cat([self.xt,self.vt],dim=-1)
        self.inputs = self.xv
        self.context = self.norm_masses.unsqueeze(1) # don't want (x0,v0) in context token
        if self.k_context:
            self.context = torch.cat([self.context,self.norm_k.unsqueeze(1)],dim=1)


    def __getitem__(self, idx):
        # Input sequence and next step
        input = self.inputs[idx,:-1]
        target = self.inputs[idx,1:]
        context = self.context[idx]
        mask = self.mask[idx]
        return input, target, context, mask


class DampedSHODatasetV2(IterableDataset):
    def __init__(self, k=(10,20), beta=(0,4), m=(1,10), x0=(-1,1), v0=(-1,1), w0=None, dt=0.1, 
                       seq_len=50, min_seq_length=20, pin_amplitude=None, min_amplitude=None, 
                       k_context=False, vary_length=False, xv=True, underdamped=False, overdamped=False):
        # some hard-coded values:
        self.max_beta = self.get_max_param(beta) if self.get_max_param(beta) is not None else 4
        self.max_k = self.get_max_param(k) if self.get_max_param(k) is not None else 20
        self.max_m = self.get_max_param(m) if self.get_max_param(m) is not None else 10
        # physical parameters
        self.k = k
        self.m = m
        self.beta = beta
        self.x0 = x0
        self.v0 = v0
        self.w0 = w0
        self.dt = dt
        self.use_w_directly = False if self.w0 is None else True
        self.underdamped = underdamped
        self.overdamped = overdamped
        if self.underdamped or self.overdamped:
            self.use_w_directly = True
        if self.underdamped and self.overdamped:
            print("Please don't ask for both underdamped AND overdamped data; just ask for neither!")

        # other parameters
        self.seq_len = seq_len
        self.min_seq_length = min_seq_length
        self.pin_amplitude = pin_amplitude
        self.min_amplitude = min_amplitude
        self.tmax = self.dt * self.seq_len
        self.k_context = k_context
        self.vary_length = vary_length
        self.xv = xv

    def get_params(self):
        beta = self.sample_param(self.beta)
        x0 = self.sample_param(self.x0)
        v0 = self.sample_param(self.v0)
        if self.use_w_directly:
            if self.underdamped:
                w0 = self.sample_param((beta,self.max_beta))
            elif self.overdamped:
                w0 = self.sample_param((0,beta))
            else:
                w0 = self.sample_param(self.w0)
            k = None
            m = None
        else:
            k = self.sample_param(self.k)
            m = self.sample_param(self.m)
            w0 = np.sqrt(k/m)
        return k,m,beta,x0,v0,w0
    
    def get_max_param(self,p):
        if isinstance(p,Iterable):
            if type(p) == tuple:
                return max(p)
            elif type(p[0]) == tuple:
                return max([max(pi) for pi in p])
            else:
                return max(p)
        else:
            return None

    def sample_param(self,p):
        if isinstance(p,Iterable):
            if type(p) == tuple:
                assert len(p) == 2
                return p[0] + np.random.rand()*(p[1]-p[0])
            elif type(p[0]) == tuple:
                assert np.all([len(pi) == 2 and type(pi) == tuple for pi in p])
                return utils.random_multiInterval(p)
            else:
                return np.random.choice(p)
        else:
            return p

    def generate_data(self):
        k,m,beta,x0,v0,w0 = self.get_params()

        # compute frequency & change v0 to fix amplitude if requested
        if self.pin_amplitude is not None:
            if self.min_amplitude is not None:
                assert self.pin_amplitude >= self.min_amplitude
            sign = np.sign(v0)
            v0 = sign * np.sqrt(w0**2*(self.pin_amplitude**2 - x0**2))
        A = np.sqrt(x0**2 + (v0/w0)**2)
        if self.min_amplitude is not None:
            if A < self.min_amplitude:
                A = self.min_amplitude
                v0 = np.sign(v0)*np.sqrt(w0**2*(A**2 - x0**2))
        
        # make time series data
        xt, vt, time = pdata2.generate_damped_harmonic_data(w0,beta,x0,v0,self.dt,self.seq_len)

        # compute mask to restrict length if desired
        if self.vary_length:
            length = np.random.randint(self.min_seq_length,self.seq_len+1)
            mask = (np.arange(self.seq_len) < length).astype(float)
        else:
            mask = -999 * np.ones(self.seq_len)

        # create context vector (legacy, not used)
        context = np.array([m,x0,v0])

        return xt, vt, time, mask, context

    def get_series(self):
        xt, vt, time, mask, context = self.generate_data()
        if self.xv:
            vec = np.concatenate([xt.reshape(-1,1),vt.reshape(-1,1)],axis=1)
        else:
            vec = xt.reshape(-1,1)
        inpt = vec[:-1,:]
        target = vec[1:,:]
        return inpt, target, context, mask


    def __iter__(self):
        while True:
            inpt, target, context, mask = self.get_series()
            yield torch.tensor(inpt, dtype=torch.float32), \
                  torch.tensor(target,dtype=torch.float32), \
                  torch.tensor(context,dtype=torch.float32), \
                  torch.tensor(mask,dtype=torch.float32).unsqueeze(-1)
    
    
    def compute_damping_status(self,k,m,beta):
        w0 = np.sqrt(k/m)
        if beta == 0:
            return 1
        elif beta > 0 and beta < w0:
            return 2
        elif beta == w0:
            return 3
        elif beta > w0:
            return 4
        else:
            print(f"Something is wrong, beta = {beta} and w0 = {w0}")
            sys.exit()