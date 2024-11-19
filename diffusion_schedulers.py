import torch
import torch.nn as nn


class UniformSampler(nn.Module):
    '''
    Deterministic sampler that samples uniformly between two points,
    X and Y, in the latent space.
    timesteps : int - number of discrete steps between X and Y
    '''
    def __init__(self, timesteps : int) -> None:
        self.timesteps = timesteps
        self.delta_t = 1.0 / timesteps
        
    def forward(self, X_T, X_0):
        t = torch.randint(1, self.timesteps+1, (X_T.shape[0],)) * self.delta_t
        X_t = X_T*t + X_0*(1-t)
        X_t_minus_1 = X_T*(t-self.delta_t) + X_0*(1-(t-self.delta_t))
        epsilon = X_t - X_t_minus_1
        return X_t, X_t_minus_1, epsilon, t
    
