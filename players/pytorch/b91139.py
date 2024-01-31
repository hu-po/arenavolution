import torch.nn as nn
from einops.layers.torch import Rearrange
from torch.nn.utils import weight_norm

class Block(nn.Module):
    def __init__(self, num_classes, in_channels=3, hidden_layers=[128, 256, 512], kernel_size=3, activation=nn.ReLU, use_weight_norm=True, dropout_prob=0.1):
        super(Block, self).__init__()

        layers = []
        for out_channels in hidden_layers:
            layers.append(
                weight_norm(nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2, bias=False))
                if use_weight_norm else
                nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2, bias=False)
            )
            layers.append(activation())
            layers.append(nn.MaxPool2d(2))
            layers.append(nn.Dropout(dropout_prob))
            in_channels = out_channels
        
        self.encoder = nn.Sequential(*layers)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            Rearrange('b c h w -> b (c h w)'),
            nn.Linear(hidden_layers[-1], num_classes)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.classifier(x)
        return x