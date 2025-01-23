import torch
from torch import nn
import numpy as np
from vq_vae.types_ import *
from vq_vae.vq_vae import VQVAE


class ConvVAE(VQVAE):
    
    def __init__(self, 
                 kl_min: float = 0.0, # minimum kl weight
                 kl_max: float = 0.1, # maximum kl weight
                 max_steps: int = 20000, # number of training steps for cosine annealing
                 *args, **kwargs) -> None:
        super(ConvVAE, self).__init__(*args, **kwargs)
        # self contained cosine annealing for the kl divergence weight
        self.kl_min = kl_min
        self.kl_weight = kl_min
        self.kl_max = kl_max
        self.max_steps = max_steps
        self.global_step = 0
        
        self.mu = nn.Conv2d(
            self.embedding_dim, self.embedding_dim, kernel_size=3, stride=1, padding=1
        )
        self.log_var = nn.Conv2d(
            self.embedding_dim, self.embedding_dim, kernel_size=3, stride=1, padding=1
        )
        
    def forward(self, x) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x1 = self.encoder(x)
        mu_z = self.mu(x1)
        log_var_z = self.log_var(x1)
        z = self.reparameterise(mu_z, log_var_z)
        kl_loss = self.kl_weight * self.kl_divergence_standard_normal(mu_z, log_var_z)
        self.step_cosine_annealing_kl_weight()
        return self.decoder(z), x, kl_loss
        #return self.decoder(x1), x, torch.zeros(1, device=x.device)
    
    def reparameterise(self, mu, log_var) -> dict:
        std = torch.exp(0.5 * log_var)
        epsilon = torch.randn_like(std)
        return mu + (epsilon * std)
    
    def kl_divergence_standard_normal(self, mu, log_var) -> torch.Tensor:
        return -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
    
    def step_cosine_annealing_kl_weight(self) -> None:
        # kl_weight is gradually increased from kl_min to kl_max during training
        if self.global_step < self.max_steps:
            self.kl_weight = self.kl_max + 0.5 * (self.kl_min - self.kl_max) * (
                1 + np.cos(np.pi * self.global_step / self.max_steps)
            )
            self.global_step += 1
