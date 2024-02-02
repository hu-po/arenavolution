import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
from torch.nn.utils import weight_norm

class Block(nn.Module):
    def __init__(self, num_classes, in_channels=3, hidden_dims=(64, 128, 256), kernel_sizes=(3, 5, 3), activation_fn=nn.ReLU, use_pooling=True, pooling_type='avg', use_weight_norm=False):
        super(Block, self).__init__()
        layers = []
        for idx, (hidden_dim, kernel_size) in enumerate(zip(hidden_dims, kernel_sizes)):
            padding = kernel_size // 2
            conv_layer = nn.Conv2d(in_channels, hidden_dim, kernel_size, stride=1, padding=padding)
            if use_weight_norm:
                conv_layer = weight_norm(conv_layer)
            layers.append(conv_layer)
            layers.append(activation_fn())
            if use_pooling:
                pool_layer = nn.MaxPool2d(2) if pooling_type == 'max' else nn.AvgPool2d(2)
                layers.append(pool_layer)
            in_channels = hidden_dim

        self.feature_extractor = nn.Sequential(*layers)
        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(hidden_dims[-1], num_classes)

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.pooling(x).view(x.size(0), -1)
        x = self.classifier(x)
        return x