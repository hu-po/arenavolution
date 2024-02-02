import torch
import torch.nn as nn
from einops.layers.torch import Rearrange

class Block(nn.Module):
    def __init__(self, num_classes, in_channels=3, hidden_channels=(128, 256, 512), kernel_size=3, activation=nn.ReLU, use_weight_norm=True, pool_type='avg', dropout_rate=0.1):
        super(Block, self).__init__()
        layers = []
        
        for out_channels in hidden_channels:
            conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=kernel_size // 2)
            if use_weight_norm:
                conv2d = nn.utils.weight_norm(conv2d)
            layers += [
                conv2d,
                activation(),
                nn.Dropout(dropout_rate),
                nn.AvgPool2d(2) if pool_type == 'avg' else nn.MaxPool2d(2)
            ]
            in_channels = out_channels

        self.feature_extractor = nn.Sequential(*layers)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_channels[-1], num_classes)
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.global_pool(x)
        x = self.flatten(x)
        x = self.classifier(x)
        return x