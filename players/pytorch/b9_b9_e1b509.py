import torch
import torch.nn as nn
from einops.layers.torch import Rearrange

class Block(nn.Module):
    def __init__(self, num_classes, in_channels=3, hidden_dims=[48, 96, 192], kernel_sizes=[3, 3, 3], strides=[1, 2, 2], activation_fn=nn.LeakyReLU, pooling='max'):
        super(Block, self).__init__()
        layers = []
        for hidden_dim, kernel_size, stride in zip(hidden_dims, kernel_sizes, strides):
            layers.extend([
                nn.Conv2d(in_channels, hidden_dim, kernel_size, stride=stride, padding=kernel_size // 2),
                activation_fn(),
                nn.MaxPool2d(kernel_size=2) if pooling == 'max' else nn.AvgPool2d(kernel_size=2)
            ])
            in_channels = hidden_dim
        
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            Rearrange('b c h w -> b (c h w)'),
            nn.Dropout(0.5),
            nn.Linear(hidden_dims[-1], num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x