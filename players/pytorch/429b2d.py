import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
from torch.nn.utils import weight_norm

class Block(nn.Module):
    def __init__(self, num_classes, in_channels=3, hidden_dims=(64, 128, 256, 512), kernel_size=3, activation_fn=nn.ReLU, use_weight_norm=True, dropout=0.25):
        super(Block, self).__init__()
        
        self.layers = nn.Sequential()
        for idx, hidden_dim in enumerate(hidden_dims):
            if use_weight_norm:
                conv = weight_norm(nn.Conv2d(in_channels=in_channels if idx == 0 else hidden_dims[idx - 1],
                                             out_channels=hidden_dim,
                                             kernel_size=kernel_size,
                                             stride=1,
                                             padding=kernel_size // 2))
            else:
                conv = nn.Conv2d(in_channels=in_channels if idx == 0 else hidden_dims[idx - 1],
                                 out_channels=hidden_dim,
                                 kernel_size=kernel_size,
                                 stride=1,
                                 padding=kernel_size // 2)
            
            self.layers.add_module(f"conv{idx}", conv)
            self.layers.add_module(f"act{idx}", activation_fn())
            self.layers.add_module(f"pool{idx}", nn.MaxPool2d(kernel_size=2))
            if dropout > 0.0:
                self.layers.add_module(f"drop{idx}", nn.Dropout(dropout))
        
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            Rearrange('b c h w -> b (c h w)'),
            nn.Linear(hidden_dims[-1], num_classes)
        )

    def forward(self, x):
        x = self.layers(x)
        x = self.classifier(x)
        return x