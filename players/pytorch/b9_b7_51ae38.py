import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
from torch.nn.utils import weight_norm

class Block(nn.Module):
    def __init__(self, num_classes, in_channels=3, hidden_dims=[64, 128, 256], kernel_size=3, stride=1, padding=1):
        super(Block, self).__init__()
        layers = []
        for hidden_dim in hidden_dims:
            layers += [
                weight_norm(nn.Conv2d(in_channels, hidden_dim, kernel_size=kernel_size, stride=stride, padding=padding)),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
            ]
            in_channels = hidden_dim
        
        self.feature_extractor = nn.Sequential(*layers)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dims[-1], num_classes)
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        x = x.mean([-2, -1])
        x = self.classifier(x)
        return x