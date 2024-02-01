import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
from torch.nn.utils import weight_norm

class Block(nn.Module):
    def __init__(self, num_classes, in_channels=3, hidden_dims=(128, 256), activation_fn=nn.ReLU, use_weight_norm=False, dropout=0.1):
        super(Block, self).__init__()
        
        layers = []
        for hidden_dim in hidden_dims:
            if use_weight_norm:
                layers.append(weight_norm(nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=1, padding=1)))
            else:
                layers.append(nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=1, padding=1))
            
            layers.append(activation_fn())
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            layers.append(nn.Dropout(dropout))
            in_channels = hidden_dim

        self.features = nn.Sequential(*layers)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            Rearrange('b c h w -> b (c h w)'),
            nn.Linear(hidden_dims[-1], num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x