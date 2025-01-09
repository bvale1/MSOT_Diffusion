import torch
import torch.nn as nn
import torch.nn.functional as F
from random import random


class BridgingDiffusion(nn.Module):
    '''
    Deterministic sampler that samples uniformly between two points,
    X and Y, in the latent space.
    timesteps : int - number of discrete steps between X and Y
    '''
    def __init__(self,
                 model : nn.Module,
                 training_timesteps : int
                 ) -> None:
        super().__init__()
        self.model = model
        
        self.self_condition = self.model.self_condition
        self.image_condition = self.model.image_condition
        self.chaneels = self.model.channels
        
        self.training_timesteps = training_timesteps
        self.training_delta_t = 1.0 / training_timesteps
        
    def v(self, X_T, X_0):
        return (X_T - X_0) / self.training_timesteps
        
    def pred_X_0(self, X_t, v, t):
        return X_t - v*t*self.training_delta_t
    
    def get_X_t(self, X_T, v, t):
        return X_T - v*t*self.training_delta_t
    
    def forward(self, X_T, X_0, x_cond=None):
        b, device = X_T.shape[0], X_T.device
        v = self.v(X_T, X_0)
        t = torch.randint(0, self.training_timesteps, (b,), device=device).long()
        X_t = self.get_X_t(X_T, v, t)
        
        x_self_cond = None
        if self.self_condition and random() < 0.5:
            with torch.no_grad():
                x_self_cond = self.pred_X_0(
                    X_t, self.model(X_t, t), t
                )
                x_self_cond.detach_()
                        
        v_pred = self.model(X_t, t, x_self_cond=x_self_cond, x_cond=x_cond)
        return F.mse_loss(v_pred, v)
    
    def sample(self, X_T, x_cond=None):
        b, device = X_T.shape[0], X_T.device
        for t in reversed(range(self.training_timesteps)):
            batched_times = torch.full((b,), t, device = device, dtype = torch.long)
            v_pred = self.model(X_T, batched_times, x_cond=x_cond)
            X_t = X_T - v_pred* self.training_delta_t