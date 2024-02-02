import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
from torch.nn.utils import weight_norm

class Block(nn.Module):
    def __init__(self, num_classes, in_channels=3, hidden_dims=[64, 128, 256, 512], kernel_sizes=[3, 3, 3, 3], activation=nn.GELU, use_weight_norm=True, dropout=0.3):
        super(Block, self).__init__()
        self.layers = nn.ModuleList()

        for i, (hidden_dim, kernel_size) in enumerate(zip(hidden_dims, kernel_sizes)):
            conv = nn.Conv2d(in_channels, hidden_dim, kernel_size, stride=2, padding=kernel_size // 2)
            if use_weight_norm:
                conv = weight_norm(conv)
            self.layers.append(conv)
            self.layers.append(activation())
            self.layers.append(nn.Dropout(dropout))
            in_channels = hidden_dim

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            Rearrange('b c h w -> b (c h w)'),
            nn.Linear(hidden_dims[-1], num_classes)
        )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = self.head(x)
        return x