import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
from torch.nn.utils import weight_norm

class Block(nn.Module):
    def __init__(self, num_classes, in_channels=3, hidden_layers=[256, 512], kernel_size=3, activation=nn.ReLU, use_weight_norm=True, pool_size=2, dropout_rate=0.1):
        super(Block, self).__init__()
        
        layers = []
        for out_channels in hidden_layers:
            conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2)
            if use_weight_norm:
                conv = weight_norm(conv)
            layers.extend([
                conv,
                activation(),
                nn.MaxPool2d(pool_size),
                nn.Dropout(dropout_rate)
            ])
            in_channels = out_channels

        self.features = nn.Sequential(*layers)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            Rearrange('b c 1 1 -> b c'),
            nn.Linear(hidden_layers[-1], num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x