import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
from torch.nn.utils import weight_norm

class Block(nn.Module):
    def __init__(self, num_classes, in_channels=3, hidden_dims=(64, 128, 256), kernel_sizes=(3, 5, 3), activation=nn.ReLU, use_weight_norm=True, dropout_rate=0.3, pooling_type='avg'):
        super(Block, self).__init__()

        self.layers = nn.ModuleList()
        for out_channels, kernel_size in zip(hidden_dims, kernel_sizes):
            conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=kernel_size // 2)
            if use_weight_norm:
                conv = weight_norm(conv)
            self.layers.append(nn.Sequential(
                conv,
                activation(),
                nn.Dropout(dropout_rate),
                nn.MaxPool2d(2)
            ))
            in_channels = out_channels
        
        self.pool = nn.AdaptiveAvgPool2d(1) if pooling_type == 'avg' else nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            Rearrange('b c h w -> b (c h w)'),
            nn.Linear(hidden_dims[-1], num_classes)
        )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = self.pool(x)
        x = self.fc(x)
        return x