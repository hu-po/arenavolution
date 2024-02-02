import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
from torch.nn.utils import weight_norm

class Block(nn.Module):
    def __init__(self, num_classes, in_channels=3, hidden_channels=(128, 256, 512), kernel_size=3, activation=nn.ReLU, use_weight_norm=True, pool_type='avg', dropout_rate=0.1):
        super(Block, self).__init__()
        layers = []
        
        for out_channels in hidden_channels:
            conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=kernel_size // 2)
            layers += [
                weight_norm(conv2d) if use_weight_norm else conv2d,
                activation(),
                nn.Dropout(dropout_rate),
                nn.AvgPool2d(2) if pool_type == 'avg' else nn.MaxPool2d(2)
            ]
            in_channels = out_channels

        self.feature_extractor = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = Rearrange('b c h w -> b (c h w)')
        self.classifier = nn.Linear(hidden_channels[-1], num_classes)

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.classifier(x)
        return x