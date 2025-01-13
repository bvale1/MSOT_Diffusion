import torch
import torch.nn as nn
import torch.nn.functional as F
from random import random


class BridgingDiffusion(nn.Module):
    '''
    Deterministic sampler that samples uniformly between two points,
    X and Y, in the latent space.
    model : nn.Module - model to predict the velocity of the diffusion
    training_timesteps : int - number of discrete steps between X and Y used in training
    sampling_timesteps : int - number of discrete steps between X and Y used in sampling
    '''
    def __init__(self,
                 model : nn.Module,
                 training_timesteps : int,
                 sampling_timesteps : int,
                 integration_scheme : str='Euler') -> None:
        super().__init__()
        self.model = model
        
        self.self_condition = self.model.self_condition
        self.image_condition = self.model.image_condition
        self.chaneels = self.model.channels
        
        assert integration_scheme in ['Euler', 'RK4'],  \
            'Integration scheme must be either Euler or RK4'
        self.integration_scheme = integration_scheme
        
        # training timestep is only necessary when self-conditioning
        self.training_timesteps = training_timesteps
        self.training_delta_t = 1.0 / training_timesteps
        self.sampling_timesteps = sampling_timesteps
        self.sampling_delta_t = 1.0 / sampling_timesteps
        
    def v(self, x_T, x_0):
        return x_T - x_0
    
    def x_t(self, x_0, v, t, delta_t):
        return x_0 + (v * t.view(-1,1,1,1) * delta_t)
    
    def pred_x_0(self, x_t, v_pred, t, delta_t):
        return x_t - (v_pred * t.view(-1,1,1,1) * delta_t)
        
    def forward(self, x_T, x_0, x_cond=None):
        b, device = x_T.shape[0], x_T.device
        v = self.v(x_T, x_0)
        t = torch.randint(0, self.training_timesteps, (b,), device=device).long()
        x_t = self.x_t(x_0, v, t, self.training_delta_t)
        
        x_self_cond = None
        if self.self_condition and random() < 0.5:
            with torch.no_grad():
                x_self_cond = self.pred_x_0(x_T, self.model(x_t, t))
                x_self_cond.detach_()
                        
        v_pred = self.model(x_t, t, x_self_cond=x_self_cond, x_cond=x_cond)
        return F.mse_loss(v_pred, v)
    
    def Euler_step_x_t(self, x_t, v_pred):
        return x_t - v_pred * self.sampling_delta_t
    
    def RK4_step_x_t(self, x_t, t, v_pred, self_cond=None, x_cond=None):
        k1 = v_pred
        k2 = self.model(
            x_t - k1 * self.sampling_delta_t / 2,
            t - self.sampling_timesteps / 2,
            x_self_cond=self_cond, x_cond=x_cond
        )
        k3 = self.model(
            x_t - k2 * self.sampling_delta_t / 2,
            t - self.sampling_timesteps / 2,
            x_self_cond=self_cond, x_cond=x_cond
        )
        k4 = self.model(
            x_t - k3 * self.sampling_delta_t,
            t - self.sampling_timesteps,
            x_self_cond=self_cond, x_cond=x_cond
        )
        return x_t - ((k1+(2*k2)+(2*k3)+k4)*self.sampling_delta_t/6)
    
    @torch.inference_mode()
    def sample(self, x_T, x_cond=None):
        b, device = x_T.shape[0], x_T.device
        self_cond = None
        x_t = x_T
        
        for t in reversed(range(self.sampling_timesteps)):
            batched_times = torch.full((b,), t, device=device, dtype=torch.long)
            v_pred = self.model(
                x_t, batched_times, x_self_cond=self_cond, x_cond=x_cond
            )
            match self.integration_scheme:
                case 'Euler':
                    x_t = self.Euler_step_x_t(x_t, v_pred)
                case 'RK4':
                    x_t = self.RK4_step_x_t(
                        x_t, batched_times, v_pred, self_cond=self_cond, x_cond=x_cond
                    )
            if self.self_condition:
                self_cond = self.pred_x_0(x_T, v_pred)
        
        return x_t