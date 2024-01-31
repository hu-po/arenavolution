import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
from torch.nn.utils import weight_norm

class Block(nn.Module):
    def __init__(self, num_classes, in_channels=3, hidden_dims=None, kernel_size=3, activation_fn=nn.ReLU, use_weight_norm=True, dropout_rate=0.1):
        super(Block, self).__init__()
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256]

        layers = []
        for hidden_dim in hidden_dims:
            layers.append(nn.Sequential(
                weight_norm(nn.Conv2d(in_channels, hidden_dim, kernel_size, padding=kernel_size // 2)) if use_weight_norm else nn.Conv2d(in_channels, hidden_dim, kernel_size, padding=kernel_size // 2),
                activation_fn(),
                nn.MaxPool2d(2),
                nn.Dropout(dropout_rate)
            ))
            in_channels = hidden_dim

        self.feature_extractor = nn.Sequential(*layers)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            Rearrange('b c h w -> b (c h w)'),
            nn.Linear(hidden_dims[-1], num_classes)
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.classifier(x)
        return x