import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
from torch.nn.utils import weight_norm

class Block(nn.Module):
    def __init__(self, num_classes, in_channels=3, hidden_dims=None, kernel_size=3, activation_fn=nn.ReLU, use_weight_norm=True, dropout=0.1):
        super(Block, self).__init__()
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256]

        layers = []
        for i, hidden_dim in enumerate(hidden_dims):
            layers.append(nn.Conv2d(in_channels if i == 0 else hidden_dims[i-1], hidden_dim, kernel_size, padding=kernel_size//2))
            if use_weight_norm:
                layers[-1] = weight_norm(layers[-1])
            layers.append(activation_fn(inplace=True))
            layers.append(nn.MaxPool2d(2))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))

        self.features = nn.Sequential(*layers)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            Rearrange('b c h w -> b (c h w)'),
            nn.Linear(hidden_dims[-1], num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x