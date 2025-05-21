import torch
import torch.nn as nn


class VAE(nn.Module):
    
    def __init__(self,
                 encoder : nn.Module,
                 decoder : nn.Module,
                 ) -> None:
        super().__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        
    def forward(self, x):
        mu_z, log_var_z = self.encoder(x)
        z = self.reparameterise(mu_z, log_var_z)
        return self.decoder(z), mu_z, log_var_z
    
    def reparameterise(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        epsilon = torch.randn_like(std)
        return mu + (epsilon * std)


class AutoEncoder(nn.Module):
    def __init__(self,
                 encoder : nn.Module,
                 decoder : nn.Module,
                 ) -> None:
        super().__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        
    def forward(self, x):
        z, _ = self.encoder(x)
        return self.decoder(z)