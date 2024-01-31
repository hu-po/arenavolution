import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
from torch.nn.utils import weight_norm

class Block(nn.Module):
    def __init__(self, num_classes, in_channels=3, hidden_dims=(64, 128, 256, 512), kernel_size=3, activation_fn=nn.ReLU, use_weight_norm=True, dropout=0.3):
        super(Block, self).__init__()
        
        layers = []
        for idx, hidden_dim in enumerate(hidden_dims[:-1]):
            layers.extend(self._conv_block(in_channels if idx == 0 else hidden_dims[idx - 1], hidden_dim, kernel_size, activation_fn, use_weight_norm, dropout))
            layers.append(nn.MaxPool2d(kernel_size=2))

        # Last layer without pooling
        layers.extend(self._conv_block(hidden_dims[-2], hidden_dims[-1], kernel_size, activation_fn, use_weight_norm, dropout))

        self.features = nn.Sequential(*layers)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            Rearrange('b c h w -> b (c h w)'),
            nn.Linear(hidden_dims[-1], num_classes),
        )

    def _conv_block(self, in_channels, out_channels, kernel_size, activation_fn, use_weight_norm, dropout):
        layers = [
            weight_norm(nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size//2)) if use_weight_norm else nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size//2),
            activation_fn(),
            nn.Dropout(dropout)
        ]
        return layers

    def forward(self, x):
        x = self.features(x)
        x = self.head(x)
        return x