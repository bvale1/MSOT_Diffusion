import torch, wandb
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from typing import Optional

from pytorch_utils import KlDivergenceStandaredNormal

class ResidualBlock(nn.Module):
    def __init__(self, channels : int, 
                 kernel_size : int=3, 
                 stride : int=1, 
                 padding : int=1,
                 *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.channels = channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        self.doubleconv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size, stride, padding),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size, stride, padding),
            nn.BatchNorm2d(channels),
            nn.ReLU()
        )
        
    def forward(self, x):
        return x + self.doubleconv(x)
    

class DownsampleBlock(nn.Module):
    def __init__(self, in_channels : int,
                 out_channels : int, 
                 kernel_size : int=3, 
                 stride : int=1, 
                 padding : int=1,
                 *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        self.inconv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding
        )
        self.residual = ResidualBlock(
            out_channels, kernel_size, stride, padding
        )
        self.downsample = nn.AvgPool2d(2)
        
    def forward(self, x):
        x = self.inconv(x)
        x = self.residual(x)
        return self.downsample(x)

    
class Encoder(nn.Module):
    def __init__(self,
                 input_size : tuple=(256, 256),
                 input_channels : int=1,
                 hidden_channels : list=[32, 64, 128, 256, 512, 1024],
                 latent_dim : int=1024,
                 *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.latent_channels = latent_dim
        
        self.downsample1 = DownsampleBlock(input_channels, hidden_channels[0])
        for i in range(len(hidden_channels)-1):
            setattr(self, f'downsample{i+2}', 
                    DownsampleBlock(hidden_channels[i], hidden_channels[i+1]))
        self.fc_in_size = hidden_channels[-1] * (
            (input_size[0] // 2**(len(hidden_channels))) * 
            (input_size[1] // 2**(len(hidden_channels)))
        )
        self.mu_z_fc = nn.Linear(self.fc_in_size, latent_dim)
        self.log_var_z_fc = nn.Linear(self.fc_in_size, latent_dim)
        
    def forward(self, x):
        x = self.downsample1(x)
        for i in range(1, len(self.hidden_channels)):
            x = getattr(self, f'downsample{i+1}')(x)
        x = x.view(x.size(0), -1) # (batch_size, 1024, a, b) -> (batch_size, 1024*a*b)
        return self.mu_z_fc(x), self.log_var_z_fc(x)
    

class UpsampleBlock(nn.Module):
    def __init__(self,
                 in_channels : int,
                 out_channels : int, 
                 kernel_size : int=2, 
                 stride : int=2, 
                 padding : int=0
                 ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        self.upsample = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size, stride, padding
        )
        self.residual = ResidualBlock(
            out_channels, kernel_size=3, stride=1, padding=1
        )
        
    def forward(self, x):
        x = self.upsample(x)
        return self.residual(x)


class Decoder(nn.Module):
    def __init__(self,
                 hidden_channels : list=[1024, 512, 256, 128, 64, 32],
                 output_channels : int=1,
                 output_size : tuple=(256, 256),
                 latent_dim : int=1024,
                 *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.hidden_channels = hidden_channels
        self.output_channels = output_channels
        self.output_size = output_size
        self.latent_dim = latent_dim
        
        # upsample the latent vector to ensure that the output tensor will match self.output_size
        self.fc_out_size = hidden_channels[0] * (
            (output_size[0] // 2**(len(hidden_channels))) * 
            (output_size[1] // 2**(len(hidden_channels)))
        )
        self.fc_out_reshape = (hidden_channels[0],
                               output_size[0] // 2**(len(hidden_channels)), 
                               output_size[1] // 2**(len(hidden_channels)))
        self.fc1 = nn.Sequential(
            nn.Linear(latent_dim, self.fc_out_size), nn.ReLU()
        )
        self.upconv1 = nn.ConvTranspose2d(
            hidden_channels[0], hidden_channels[0], kernel_size=2, stride=2
        )
        for i in range(len(hidden_channels)-1):
            setattr(
                self,
                f'upsampleblock{i+1}',
                UpsampleBlock(hidden_channels[i], hidden_channels[i+1])
            )
        self.outconv = nn.Conv2d(hidden_channels[-1], output_channels, kernel_size=1)
        
    def forward(self, x):
        x = self.fc1(x)
        x = x.view(
            x.size(0),
            self.fc_out_reshape[0],
            self.fc_out_reshape[1],
            self.fc_out_reshape[2]
        )
        x = self.upconv1(x)
        for i in range(len(self.hidden_channels)-1):
            x = getattr(self, f'upsampleblock{i+1}')(x)
        return self.outconv(x)
    

class LightningVAE(pl.LightningModule):
    def __init__(self,
                 encoder : nn.Module,
                 decoder : nn.Module,
                 kl_weight : float=1.0,
                 wandb_log : Optional[wandb.sdk.wandb_run.Run] = None, # wandb logger
                 git_hash : Optional[str] = None, # git hash of the current commit
                 lr : Optional[float] = 1e-3, # learning rate
                 seed : int = None # seed for reproducibility
                 ) -> None:
        super().__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.kl_weight = kl_weight
        
        self.wandb_log = wandb_log
        self.git_hash = git_hash
        self.lr = lr
        self.seed = seed
        self.mse_loss = nn.MSELoss(reduction='mean')
        self.kl_loss = KlDivergenceStandaredNormal()
        self.save_hyperparameters(ignore=['encoder', 'decoder'])
        
    def forward(self, x):
        mu_z, log_var_z = self.encoder(x)
        z = self.reparameterise(mu_z, log_var_z)
        return self.decoder(z), mu_z, log_var_z
    
    def reparameterise(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        epsilon = torch.randn_like(std)
        return mu + (epsilon * std)
    
    def training_step(self, batch, batch_idx):
        x = batch[0]
        x_hat, mu_z, log_var_z = self.forward(x)
        mse = self.mse_loss(x_hat, x)
        kl = self.kl_weight * self.kl_loss(mu_z, log_var_z)
        loss = mse + kl
        if self.wandb_log:
            self.logger.experiment.log(
                {'train_loss' : loss, 'train_mse' : mse, 'train_kl' : kl},
                step=self.trainer.global_step
            )
        return loss
    
    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            x = batch[0]
            x_hat, mu_z, log_var_z = self.forward(x)
            mse = self.mse_loss(x_hat, x)
            kl = self.kl_weight * self.kl_loss(mu_z, log_var_z)
            loss = mse + kl
            self.log('val_loss', loss)
            if self.wandb_log:
                self.logger.experiment.log(
                    {'val_loss' : loss, 'val_mse' : mse, 'val_kl' : kl},
                    step=self.trainer.global_step
                )
            return loss
    
    def test_step(self, batch, batch_idx):
        with torch.no_grad():
            x = batch[0]
            x_hat, mu, log_var = self.forward(x)
            mse = self.mse_loss(x_hat, x)
            kl = self.kl_weight * self.kl_loss(mu, log_var)
            loss = mse + kl
            if self.wandb_log:
                self.logger.experiment.log(
                    {'test_loss' : loss, 'test_mse' : mse, 'test_kl' : kl},
                    step=self.trainer.global_step
                )
            return loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
    
    
class LightningAutoEncoder(pl.LightningModule):
    def __init__(self,
                 encoder : nn.Module,
                 decoder : nn.Module,
                 wandb_log : Optional[wandb.sdk.wandb_run.Run] = None, # wandb logger
                 git_hash : Optional[str] = None, # git hash of the current commit
                 lr : Optional[float] = 1e-3, # learning rate
                 seed : int = None # seed for reproducibility
                 ) -> None:
        super().__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        
        self.wandb_log = wandb_log
        self.git_hash = git_hash
        self.lr = lr
        self.seed = seed
        self.mse_loss = nn.MSELoss(reduction='mean')
        self.save_hyperparameters(ignore=['encoder', 'decoder'])
        
    def forward(self, x):
        z, _ = self.encoder(x)
        return self.decoder(z)
    
    
    def training_step(self, batch, batch_idx):
        x = batch[0]
        x_hat = self.forward(x)
        mse = self.mse_loss(x_hat, x)
        loss = mse
        if self.wandb_log:
            self.logger.experiment.log(
                {'train_loss' : loss, 'train_mse' : mse},
                step=self.trainer.global_step
            )
        return loss
    
    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            x = batch[0]
            x_hat = self.forward(x)
            mse = self.mse_loss(x_hat, x)
            loss = mse
            self.log('val_loss', loss)
            if self.wandb_log:
                self.logger.experiment.log(
                    {'val_loss' : loss, 'val_mse' : mse},
                    step=self.trainer.global_step
                )
            return loss
    
    def test_step(self, batch, batch_idx):
        with torch.no_grad():
            x = batch[0]
            x_hat = self.forward(x)
            mse = self.mse_loss(x_hat, x)
            loss = mse
            if self.wandb_log:
                self.logger.experiment.log(
                    {'test_loss' : loss, 'test_mse' : mse},
                    step=self.trainer.global_step
                )
            return loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
    