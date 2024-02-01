import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
from torch.nn.utils import weight_norm

class Block(nn.Module):
    def __init__(self, num_classes, in_channels=3, hidden_layers=[256, 512], kernel_size=3, activation=nn.ReLU, use_weight_norm=True, pool_size=2):
        super(Block, self).__init__()
        
        layers = []
        for out_channels in hidden_layers:
            conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2)
            if use_weight_norm:
                conv = weight_norm(conv)
            layers.append(conv)
            layers.append(activation())
            layers.append(nn.MaxPool2d(pool_size))
            in_channels = out_channels

        self.features = nn.Sequential(*layers)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = Rearrange('b c h w -> b (c h w)')
        self.classifier = nn.Linear(hidden_layers[-1], num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.avg_pool(x)
        x = self.flatten(x)
        x = self.classifier(x)
        return x