import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
from torch.nn.utils import weight_norm

class Block(nn.Module):
    def __init__(self, num_classes, in_channels=3, hidden_dims=[256, 512], dropout_rate=0.1, use_weight_norm=True, kernel_sizes=[3, 3], strides=[1, 1]):
        super(Block, self).__init__()
        layers = []

        for hidden_dim, kernel_size, stride in zip(hidden_dims, kernel_sizes, strides):
            conv = nn.Conv2d(in_channels, hidden_dim, kernel_size, stride=stride, padding=kernel_size // 2)
            if use_weight_norm:
                conv = weight_norm(conv)
            layers.extend([
                conv,
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.MaxPool2d(kernel_size=2)
            ])
            in_channels = hidden_dim

        self.feature_extractor = nn.Sequential(*layers)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = Rearrange('b c h w -> b (c h w)')
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dims[-1], num_classes)
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.avg_pool(x)
        x = self.flatten(x)
        x = self.classifier(x)
        return x