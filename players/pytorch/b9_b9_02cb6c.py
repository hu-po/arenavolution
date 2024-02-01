import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
from torch.nn.utils import weight_norm

class Block(nn.Module):
    def __init__(self, num_classes, in_channels=3, hidden_layers=(256, 512, 1024), kernel_size=3, activation_fn=nn.ReLU, use_weight_norm=True, dropout_rate=0.2):
        super(Block, self).__init__()
        
        layers = []
        for idx, hidden_layer in enumerate(hidden_layers):
            conv_layer = weight_norm(nn.Conv2d(in_channels, hidden_layer, kernel_size, padding=kernel_size // 2)) if use_weight_norm else nn.Conv2d(in_channels, hidden_layer, kernel_size, padding=kernel_size // 2)
            layers.append(conv_layer)
            layers.append(activation_fn())
            if idx < len(hidden_layers) - 1:
                layers.append(nn.MaxPool2d(kernel_size=2))
                layers.append(nn.Dropout(dropout_rate))
            in_channels = hidden_layer

        self.feature_extractor = nn.Sequential(*layers)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            Rearrange('b c h w -> b (c h w)'),
            nn.Linear(hidden_layers[-1], num_classes),
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.classifier(x)
        return x