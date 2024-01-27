import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
from torch.nn.utils import weight_norm

class Block(nn.Module):
    def __init__(self, num_classes, in_channels=3, hidden_dims=None, kernel_size=3, activation_fn=nn.ReLU, use_weight_norm=True, dropout=0.25):
        super(Block, self).__init__()
        if hidden_dims is None:
            hidden_dims = [64, 128, 256, 512]
        
        layers = []
        for idx, hidden_dim in enumerate(hidden_dims):
            if idx > 0:
                layers.append(nn.MaxPool2d(kernel_size=2))
            
            conv_layer = nn.Conv2d(in_channels=in_channels if idx == 0 else hidden_dims[idx-1],
                                   out_channels=hidden_dim,
                                   kernel_size=kernel_size,
                                   padding=kernel_size // 2)
            if use_weight_norm:
                conv_layer = weight_norm(conv_layer)
            layers.append(conv_layer)
            layers.append(activation_fn(inplace=True))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            Rearrange('b c h w -> b (c h w)'),
            nn.Linear(hidden_dims[-1], num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x