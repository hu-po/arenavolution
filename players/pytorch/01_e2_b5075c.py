import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
from torch.nn.utils import weight_norm

class Block(nn.Module):
    def __init__(self, num_classes, in_channels=3, hidden_dims=(32, 64, 128, 256), kernel_size=3, activation_fn=nn.GELU, use_weight_norm=True, dropout=0.2):
        super(Block, self).__init__()

        layers = []
        for i, hidden_dim in enumerate(hidden_dims):
            if use_weight_norm:
                conv = weight_norm(nn.Conv2d(in_channels, hidden_dim, kernel_size, padding=kernel_size // 2))
            else:
                conv = nn.Conv2d(in_channels, hidden_dim, kernel_size, padding=kernel_size // 2)
            in_channels = hidden_dim
            
            layers.extend([
                conv,
                activation_fn(),
                nn.Dropout(dropout)
            ])

        self.encoder = nn.Sequential(*layers)
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            Rearrange('b c h w -> b (c h w)'),
        )
        self.classifier = nn.Linear(hidden_dims[-1], num_classes)

    def forward(self, x):
        x = self.encoder(x)
        x = self.global_pool(x)
        x = self.classifier(x)
        return x