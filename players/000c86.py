import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
from torch.nn.utils import weight_norm

class Block(nn.Module):
    def __init__(self, num_classes, in_channels=3, hidden_dims=None, kernel_sizes=None, activation_fn=nn.ReLU, use_weight_norm=True, dropout=0.5):
        super(Block, self).__init__()
        if hidden_dims is None:
            hidden_dims = [64, 128, 256, 512]
        if kernel_sizes is None:
            kernel_sizes = [3, 3, 3, 3]

        self.blocks = nn.ModuleList()
        for i, (hidden_dim, kernel_size) in enumerate(zip(hidden_dims, kernel_sizes)):
            conv_layer = nn.Conv2d(in_channels if i == 0 else hidden_dims[i - 1], hidden_dim, kernel_size, stride=1, padding=kernel_size // 2)
            if use_weight_norm:
                conv_layer = weight_norm(conv_layer)
            block_layers = [
                conv_layer,
                activation_fn(inplace=True),
                nn.MaxPool2d(2)
            ]
            if dropout > 0:
                block_layers.append(nn.Dropout(dropout))
            self.blocks.append(nn.Sequential(*block_layers))

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = Rearrange('b c h w -> b (c h w)')
        self.fc = nn.Linear(hidden_dims[-1], num_classes)

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        x = self.global_pool(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x