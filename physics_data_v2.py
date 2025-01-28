import numpy as np
import sys

# Generate synthetic data: simple harmonic motion
def generate_harmonic_data(m, k, x0, v0, dt, num_samples):
    w0 = np.sqrt(k/m)
    c1 = x0
    c2 = v0/w0
    # reshape & make time series
    time = np.arange(0,num_samples)*dt
    position = c1 * np.cos(w0*time) + c2 * np.sin(w0*time)
    velocity = -w0*c1*np.sin(w0*time) + w0*c2*np.cos(w0*time)
    return position, velocity, time

def generate_damped_harmonic_data(w0, beta, x0, v0, dt, num_samples):
    w1 = np.sqrt(np.abs(w0**2 - beta**2))

    if beta == 0:
        position, velocity, time = sol_undamped(x0,v0,w0,num_samples,dt)
    elif beta > 0 and beta < w0:
        position, velocity, time = sol_underdamped(x0,v0,w1,beta,num_samples,dt)
    elif beta == w0:
        position, velocity, time = sol_critdamped(x0,v0,beta,num_samples,dt)
    elif beta > w0:
        position, velocity, time = sol_overdamped(x0,v0,w1,beta,num_samples,dt)
    else:
        print(f"Error: cannot generate harmonic data with parameters w0 = {w0}, beta = {beta}, x0 = {x0}, v0 = {v0}, dt = {dt}")
        sys.exit()

    return position, velocity, time

def sol_undamped(x0,v0,w0,num_samples,dt):
    c1 = x0
    c2 = v0/w0
    time = np.arange(0,num_samples)*dt
    position = c1 * np.cos(w0*time) + c2 * np.sin(w0*time)
    velocity = -w0*c1*np.sin(w0*time) + w0*c2*np.cos(w0*time)
    return position, velocity, time

def sol_underdamped(x0,v0,w1,beta,num_samples,dt):
    c1 = x0
    c2 = (v0+beta*x0)/w1
    c1_v = v0
    c2_v = -w1*x0 - beta*(v0 + beta*x0)/w1
    time = np.arange(0,num_samples) * dt
    position = np.exp(-beta*time)*(c1*np.cos(w1*time) + c2*np.sin(w1*time))
    velocity = np.exp(-beta*time)*(c1_v*np.cos(w1*time) + c2_v*np.sin(w1*time))
    position = np.nan_to_num(position,nan=0.0)
    velocity = np.nan_to_num(velocity,nan=0.0)
    return position, velocity, time

def sol_critdamped(x0,v0,beta,num_samples,dt):
    c1 = x0
    c2 = v0 + beta*x0
    c1_v = v0
    c2_v = -beta*(v0 + beta*x0)
    time = np.arange(0,num_samples*dt,dt)
    position = c1*np.exp(-beta*time) + c2*time*np.exp(-beta*time)
    velocity = c1_v*np.exp(-beta*time) + c2_v*time*np.exp(-beta*time)
    position = np.nan_to_num(position,nan=0.0)
    velocity = np.nan_to_num(velocity,0.0)
    return position, velocity, time

def sol_overdamped(x0,v0,w1,beta,num_samples,dt):
    bminus = beta - w1
    bplus = beta + w1
    c1 = (v0 + bplus*x0)/(2*w1)
    c2 = -(v0 + bminus*x0)/(2*w1)
    c1_v = -bminus*c1
    c2_v = -bplus*c2
    time = np.arange(0,num_samples*dt,dt)
    position = c1*np.exp(-bminus*time) + c2*np.exp(-bplus*time)
    velocity = c1_v*np.exp(-bminus*time) + c2_v*np.exp(-bplus*time)
    position = np.nan_to_num(position,nan=0.0)
    velocity = np.nan_to_num(velocity,nan=0.0)
    return position, velocity, time