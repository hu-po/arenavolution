import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
from torch.nn.utils import weight_norm

class Block(nn.Module):
    def __init__(self, num_classes, in_channels=3, intermediate_channels=128, kernel_size=3, reduction_ratio=4, use_weight_norm=True):
        super(Block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, intermediate_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        self.act1 = nn.ReLU()
        self.conv2 = nn.Conv2d(intermediate_channels, intermediate_channels // reduction_ratio, kernel_size=1)
        self.act2 = nn.ReLU()
        self.conv3 = nn.Conv2d(intermediate_channels // reduction_ratio, intermediate_channels, kernel_size=3, padding=1)
        self.act3 = nn.ReLU()
        
        if use_weight_norm:
            self.conv1 = weight_norm(self.conv1)
            self.conv2 = weight_norm(self.conv2)
            self.conv3 = weight_norm(self.conv3)
        
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.rearrange = Rearrange('b c h w -> b (c h w)')
        self.fc = nn.Linear(intermediate_channels, num_classes)

    def forward(self, x):
        x = self.act1(self.conv1(x))
        x = self.act2(self.conv2(x))
        x = self.act3(self.conv3(x))
        x = self.avg_pool(x)
        x = self.rearrange(x)
        x = self.fc(x)
        return x