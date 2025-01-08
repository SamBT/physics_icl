import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
import numpy as np
from collections.abc import Iterable

# Generate synthetic data: simple harmonic motion
def generate_harmonic_data(m, k, x0, v0, dt, num_samples):
    assert m.shape[0] == x0.shape[0] == v0.shape[0]
    w0 = torch.sqrt(k/m)
    c1 = x0
    c2 = v0/w0
    # reshape & make time series
    c1 = c1.unsqueeze(1)
    c2 = c2.unsqueeze(1)
    w0 = w0.unsqueeze(1)
    time = torch.arange(0,num_samples*dt,dt)
    position = c1 * torch.cos(w0*time) + c2 * torch.sin(w0*time)
    velocity = -w0*c1*torch.sin(w0*time) + w0*c2*torch.cos(w0*time)
    return position, velocity, time

def generate_damped_harmonic_data(m, k, beta, x0, v0, dt, num_samples):
    #assert m.shape[0] == x0.shape[0] == v0.shape[0]
    w0 = torch.sqrt(k/m)
    w1 = torch.sqrt(np.abs(w0**2 - beta**2))
    
    mask_undamp = (beta==0).float().unsqueeze(1)
    mask_underdamp = ((beta<w0)&(beta>0)).float().unsqueeze(1)
    mask_critdamp = (beta==w0).float().unsqueeze(1)
    mask_overdamp = (beta>w0).float().unsqueeze(1)
    
    pos_undamped, vel_undamped, time = generate_harmonic_data(m,k,x0,v0,dt,num_samples)
    pos_underdamped, vel_underdamped, time = sol_underdamped(x0,v0,w1,beta,num_samples,dt)
    pos_critdamped, vel_critdamped, time = sol_critdamped(x0,v0,beta,num_samples,dt)
    pos_overdamped, vel_overdamped, time = sol_overdamped(x0,v0,w1,beta,num_samples,dt)
    
    position = pos_undamped*mask_undamp + pos_underdamped*mask_underdamp + pos_critdamped*mask_critdamp + pos_overdamped*mask_overdamp
    velocity = vel_undamped*mask_undamp + vel_underdamped*mask_underdamp + vel_critdamped*mask_critdamp + vel_overdamped*mask_overdamp

    return position, velocity, time

def sol_underdamped(x0,v0,w1,beta,num_samples,dt):
    c1 = x0
    c2 = (v0+beta*x0)/w1
    c1_v = v0
    c2_v = -w1*x0 - beta*(v0 + beta*x0)/w1
    time = torch.arange(0,num_samples*dt,dt)
    beta = beta.unsqueeze(1)
    w1 = w1.unsqueeze(1)
    c1 = c1.unsqueeze(1)
    c2 = c2.unsqueeze(1)
    c1_v = c1_v.unsqueeze(1)
    c2_v = c2_v.unsqueeze(1)
    position = torch.exp(-beta*time)*(c1*torch.cos(w1*time) + c2*torch.sin(w1*time))
    velocity = torch.exp(-beta*time)*(c1_v*torch.cos(w1*time) + c2_v*torch.sin(w1*time))
    position = torch.nan_to_num(position,nan=0.0)
    velocity = torch.nan_to_num(velocity,nan=0.0)
    return position, velocity, time

def sol_critdamped(x0,v0,beta,num_samples,dt):
    c1 = x0
    c2 = v0 + beta*x0
    c1_v = v0
    c2_v = -beta*(v0 + beta*x0)
    time = torch.arange(0,num_samples*dt,dt)
    beta = beta.unsqueeze(1)
    c1 = c1.unsqueeze(1)
    c2 = c2.unsqueeze(1)
    c1_v = c1_v.unsqueeze(1)
    c2_v = c2_v.unsqueeze(1)
    position = c1*torch.exp(-beta*time) + c2*time*torch.exp(-beta*time)
    velocity = c1_v*torch.exp(-beta*time) + c2_v*time*torch.exp(-beta*time)
    position = torch.nan_to_num(position,nan=0.0)
    velocity = torch.nan_to_num(velocity,0.0)
    return position, velocity, time

def sol_overdamped(x0,v0,w1,beta,num_samples,dt):
    bminus = beta - w1
    bplus = beta + w1
    c1 = (v0 + bplus*x0)/(2*w1)
    c2 = -(v0 + bminus*x0)/(2*w1)
    c1_v = -bminus*c1
    c2_v = -bplus*c2
    time = torch.arange(0,num_samples*dt,dt)
    bplus = bplus.unsqueeze(1)
    bminus = bminus.unsqueeze(1)
    c1 = c1.unsqueeze(1)
    c2 = c2.unsqueeze(1)
    c1_v = c1_v.unsqueeze(1)
    c2_v = c2_v.unsqueeze(1)
    position = c1*torch.exp(-bminus*time) + c2*torch.exp(-bplus*time)
    velocity = c1_v*torch.exp(-bminus*time) + c2_v*torch.exp(-bplus*time)
    position = torch.nan_to_num(position,nan=0.0)
    velocity = torch.nan_to_num(velocity,nan=0.0)
    return position, velocity, time