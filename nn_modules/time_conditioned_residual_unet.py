import math
import torch
from torch import nn


def contract(dim_in : int, dim_out : int, kernel_size : int=3) -> nn.Module:
    return nn.Sequential(
        nn.Conv2d(dim_in, dim_out, kernel_size, padding=1, stride=(2, 2)),
        nn.LeakyReLU(inplace=True)
    )

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim : int, theta : int= 10000):
        super().__init__()
        self.dim = dim
        self.theta = theta
        self.half_dim = dim // 2

    def forward(self, time):
        device = time.device
        time_emb = math.log(self.theta) / (self.half_dim - 1)
        time_emb = torch.exp(torch.arange(self.half_dim, device=device) * -time_emb)
        time_emb = time[:, None] * time_emb[None, :]
        time_emb = torch.cat((time_emb.sin(), time_emb.cos()), dim=-1)
        return time_emb


class TimeConditionedResNetBlock(nn.Module):
    
    def __init__(self, 
                 dim_in : int,
                 dim_out : int,
                 dim_time_emb : int=None,
                 kernel_size : int=3) -> None:
        super().__init__()
        
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.dim_time_emb = dim_time_emb
    
        if dim_time_emb is not None:
            self.mlp = nn.Linear(dim_time_emb, dim_out * 2)
            
        self.conv1 = nn.Conv2d(dim_in, dim_out, kernel_size=kernel_size, padding=1)
        self.act1 = nn.LeakyReLU(inplace=True)
        self.conv2 = nn.Conv2d(dim_out, dim_out, kernel_size=kernel_size, padding=1)
        self.act2 = nn.LeakyReLU(inplace=True)
        
        self.res_conv = nn.Conv2d(dim_in, dim_out, kernel_size=1) if dim_in != dim_out else nn.Identity()
        
        
    def forward(self, x : torch.Tensor, time_emb : torch.Tensor=None) -> torch.Tensor:
        h = self.conv1(x)
        h = self.act1(h)
        
        if self.mlp is not None and time_emb is not None:
            time_emb = self.mlp(time_emb)
            time_emb = time_emb.view(time_emb.size(0), time_emb.size(1), 1, 1)
            scale_shift = time_emb.chunk(2, dim=1)
            h = h * (scale_shift[0] + 1) + scale_shift[1]
        
        h = self.conv2(h)
        h = self.act2(h)
        
        return h + self.res_conv(x)
    
    
class TimeConditionedResUNet(nn.Module):
    
    def __init__(self, 
                 dim_in : int, # the input dimension
                 dim_out : int, # the output dimension
                 dim_first_layer : int=64, # the number of filters in the first layer
                 kernel_size : int=3,
                 theta_pos_emb : int=10000, # the theta value for the sinusoidal position embedding
                 self_condition : bool=False, # whether the finial prediction from the previous timestep will be used as conditional information
                 image_condition : bool=False, # whether an image will be provided as conditional information
                 dim_image_condition : int=1) -> None:
        super().__init__()
        
        self.channels = dim_in
        if self_condition and image_condition:
            self.dim_in = dim_in*2 + dim_image_condition
        elif self_condition:
            self.dim_in = dim_in*2
        elif image_condition:
            self.dim_in = dim_in + dim_image_condition
        else:
            self.dim_in = dim_in
        self.dim_out = dim_out
        self.dim_first_layer = dim_first_layer
        self.kernel_size = kernel_size
        self.self_condition = self_condition
        self.image_condition = image_condition
        self.dim_image_condition = dim_image_condition        
        
        self.dim_time_emb = dim_first_layer * 4
        self.sinusoidal_pos_emb = SinusoidalPosEmb(self.dim_time_emb, theta=theta_pos_emb)
        self.time_mlp = nn.Sequential(
            self.sinusoidal_pos_emb,
            nn.Linear(self.dim_time_emb, self.dim_time_emb),
            nn.GELU(),
            nn.Linear(self.dim_time_emb, self.dim_time_emb),
            nn.SiLU(inplace=True),
        )
        
        self.block1 = TimeConditionedResNetBlock(self.dim_in, dim_first_layer, self.dim_time_emb, kernel_size)
        self.contract1 = contract(dim_first_layer, dim_first_layer, kernel_size)
        self.block2 = TimeConditionedResNetBlock(dim_first_layer, dim_first_layer*2, self.dim_time_emb, kernel_size)
        self.contract2 = contract(dim_first_layer*2, dim_first_layer*2, kernel_size)
        self.block3 = TimeConditionedResNetBlock(dim_first_layer*2, dim_first_layer*(2**2), self.dim_time_emb, kernel_size)
        self.contract3 = contract(dim_first_layer*(2**2), dim_first_layer*(2**2), kernel_size)
        self.block4 = TimeConditionedResNetBlock(dim_first_layer*(2**2), dim_first_layer*(2**3), self.dim_time_emb, kernel_size)
        self.contract4 = contract(dim_first_layer*(2**3), dim_first_layer*(2**3), kernel_size)
        
        self.center_block = TimeConditionedResNetBlock(dim_first_layer*(2**3), dim_first_layer*(2**4), self.dim_time_emb, kernel_size)
        self.center_expand = nn.Sequential(
            nn.ConvTranspose2d(dim_first_layer*(2**4), dim_first_layer*(2**3), kernel_size=2, stride=(2, 2)),
            nn.LeakyReLU(inplace=True)
        )
        
        self.block5 = TimeConditionedResNetBlock(dim_first_layer*(2**4), dim_first_layer*(2**3), self.dim_time_emb, kernel_size)
        self.expand5 = nn.ConvTranspose2d(dim_first_layer*(2**3), dim_first_layer*(2**2), kernel_size=2, stride=(2, 2))
        self.block6 = TimeConditionedResNetBlock(dim_first_layer*(2**3), dim_first_layer*(2**2), self.dim_time_emb, kernel_size)
        self.expand6 = nn.ConvTranspose2d(dim_first_layer*(2**2), dim_first_layer*2, kernel_size=2, stride=(2, 2))
        self.block7 = TimeConditionedResNetBlock(dim_first_layer*(2**2), dim_first_layer*2, self.dim_time_emb, kernel_size)
        self.expand7 = nn.ConvTranspose2d(dim_first_layer*2, dim_first_layer, kernel_size=2, stride=(2, 2))
        self.block8 = TimeConditionedResNetBlock(dim_first_layer*2, dim_first_layer, self.dim_time_emb, kernel_size)
        
        self.out = nn.Conv2d(dim_first_layer, dim_out, kernel_size=1)
        
        
    def forward(self, 
                x : torch.Tensor,
                time : torch.Tensor,
                x_self_cond : torch.Tensor=None,
                x_cond : torch.Tensor=None) -> torch.Tensor:
        b, c, h, w = x.shape
        if self.self_condition:
            if x_self_cond is not None:
                x = torch.cat([x, x_self_cond], dim=1)
            else:
                x = torch.cat([x, torch.zeros_like(x)], dim=1)
        if self.image_condition:
            if x_cond is not None:
                x = torch.cat([x, x_cond], dim=1)
            else:
                x_cond = torch.zeros((b, self.dim_image_condition, h, w), device=x.device, dtype=x.dtype)
                x = torch.cat([x, x_cond], dim=1)
        
        time_emb = self.time_mlp(time)
        
        x1 = self.block1(x, time_emb)
        x2 = self.contract1(x1)
        x2 = self.block2(x2, time_emb)
        x3 = self.contract2(x2)
        x3 = self.block3(x3, time_emb)
        x4 = self.contract3(x3)
        x4 = self.block4(x4, time_emb)
        x_center = self.contract4(x4)
        
        x_center = self.center_block(x_center, time_emb)
        x_center = self.center_expand(x_center)
        
        concat = torch.cat([x_center, x4], dim=1)
        x5 = self.block5(concat, time_emb)
        x5 = self.expand5(x5)
        
        concat = torch.cat([x5, x3], dim=1)
        x6 = self.block6(concat, time_emb)
        x6 = self.expand6(x6)
        concat = torch.cat([x6, x2], dim=1)
        
        x7 = self.block7(concat, time_emb)
        x7 = self.expand7(x7)
        
        concat = torch.cat([x7, x1], dim=1)
        x8 = self.block8(concat, time_emb)
        
        return self.out(x8)
    
    
    def freeze_encoder(self):
        for param in self.block1():
            param.requires_grad = False
        for param in self.contract1():
            param.requires_grad = False
        for param in self.block2():
            param.requires_grad = False
        for param in self.contract2():
            param.requires_grad = False
        for param in self.block3():
            param.requires_grad = False
        for param in self.contract3():
            param.requires_grad = False
        for param in self.block4():
            param.requires_grad = False
        for param in self.contract4():
            param.requires_grad = False