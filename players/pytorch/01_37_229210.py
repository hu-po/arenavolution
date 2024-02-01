import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
from torch.nn.utils import weight_norm

class Block(nn.Module):
    def __init__(self, num_classes, in_channels=3, hidden_layers=(128, 256), kernel_size=3, dropout=0.1, use_weight_norm=True, activation_fn=nn.ReLU):
        super(Block, self).__init__()
        layers = []
        for i, hidden_layer in enumerate(hidden_layers):
            conv_layer = nn.Conv2d(in_channels=(in_channels if i == 0 else hidden_layers[i - 1]), 
                                   out_channels=hidden_layer,
                                   kernel_size=kernel_size, 
                                   padding=kernel_size // 2)
            if use_weight_norm:
                conv_layer = weight_norm(conv_layer)
            layers += [
                conv_layer,
                activation_fn(),
                nn.Dropout2d(dropout),
            ]
        self.feature_extractor = nn.Sequential(*layers)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            Rearrange('b c h w -> b (c h w)'),
            nn.Linear(hidden_layers[-1], num_classes)
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.classifier(x)
        return x