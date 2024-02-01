import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
from torch.nn.utils import weight_norm

class Block(nn.Module):
    def __init__(self, num_classes, in_channels=3, hidden_channels=[64, 128, 256], kernel_size=3, activation=nn.ReLU, use_weight_norm=True):
        super(Block, self).__init__()

        self.layers = nn.Sequential()
        for i, out_channels in enumerate(hidden_channels):
            conv = nn.Conv2d(in_channels if i == 0 else hidden_channels[i-1], out_channels, kernel_size, padding=(kernel_size//2))
            if use_weight_norm:
                conv = weight_norm(conv)
            self.layers.add_module(f'conv{i}', conv)
            self.layers.add_module(f'activation{i}', activation())
            self.layers.add_module(f'pooling{i}', nn.MaxPool2d(kernel_size=2, stride=2))

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            Rearrange('b c h w -> b (c h w)'),
            nn.Linear(hidden_channels[-1], num_classes)
        )

    def forward(self, x):
        x = self.layers(x)
        x = self.global_pool(x)
        x = self.classifier(x)
        return x