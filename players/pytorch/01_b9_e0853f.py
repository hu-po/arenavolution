import torch
import torch.nn as nn
from einops.layers.torch import Rearrange

class Block(nn.Module):
    def __init__(self, num_classes, in_channels=3, hidden_dims=[128, 256, 512], dropout_rate=0.1, 
                 use_weight_norm=False, kernel_sizes=[3, 3, 3], strides=[1, 1, 1], 
                 pool_sizes=[2, 2, 2], activation_fn=nn.ReLU):
        
        super(Block, self).__init__()
        layers = []

        for hidden_dim, kernel_size, stride, pool_size in zip(hidden_dims, kernel_sizes, strides, pool_sizes):
            conv = nn.Conv2d(in_channels, hidden_dim, kernel_size, stride=stride, padding=kernel_size // 2)
            layers.extend([
                conv,
                activation_fn(),
                nn.Dropout(dropout_rate),
                nn.MaxPool2d(kernel_size=pool_size)
            ])
            in_channels = hidden_dim

        self.features = nn.Sequential(*layers)
        
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            Rearrange('b c h w -> b (c h w)'),
            nn.Linear(hidden_dims[-1], num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x