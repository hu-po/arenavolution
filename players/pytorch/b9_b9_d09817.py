import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
from torch.nn.utils import weight_norm

class Block(nn.Module):
    def __init__(self, num_classes, in_channels=3, hidden_dims=(128, 256, 512), kernel_sizes=(3, 3, 3), activation=nn.ReLU, use_weight_norm=True, dropout=0.1, pooling='avg'):
        super(Block, self).__init__()
        
        layers = []
        for hidden_dim, kernel_size in zip(hidden_dims, kernel_sizes):
            conv = nn.Conv2d(in_channels, hidden_dim, kernel_size, stride=1, padding=kernel_size//2, bias=not use_weight_norm)
            if use_weight_norm:
                conv = weight_norm(conv)
            layers += [conv, activation(), nn.Dropout(dropout), nn.MaxPool2d(kernel_size=2, stride=2)]
            in_channels = hidden_dim

        self.features = nn.Sequential(*layers)

        if pooling == 'avg':
            self.pool = nn.AdaptiveAvgPool2d(1)
        elif pooling == 'max':
            self.pool = nn.AdaptiveMaxPool2d(1)
            
        self.classifier = nn.Sequential(
            Rearrange('b c h w -> b (c h w)'),
            nn.Linear(hidden_dims[-1], num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = self.classifier(x)
        return x