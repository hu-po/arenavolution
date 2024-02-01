import torch
import torch.nn as nn
from torch.nn.utils import weight_norm

class Block(nn.Module):
    def __init__(self, num_classes, in_channels=3, depths=[64, 128, 256, 512], kernel_size=3, activation=nn.ReLU, dropout=0.3, use_weight_norm=True):
        super(Block, self).__init__()
        self.layers = nn.ModuleList()
        for depth in depths:
            conv_layer = nn.Conv2d(in_channels, depth, kernel_size, padding=kernel_size//2)
            if use_weight_norm:
                conv_layer = weight_norm(conv_layer)
            layers = [
                conv_layer,
                activation(),
                nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
                nn.MaxPool2d(2)
            ]
            self.layers.append(nn.Sequential(*layers))
            in_channels = depth

        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(depths[-1], num_classes)
        )

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = self.adaptive_pool(x)
        x = self.classifier(x)
        return x