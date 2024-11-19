import torch.nn as nn


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
                 hidden_channels : list=[32, 64, 128, 256, 512, 512],
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
                 hidden_channels : list=[512, 512, 256, 128, 64, 32],
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