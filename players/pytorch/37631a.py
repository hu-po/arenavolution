import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
from torch.nn.utils import weight_norm

class Block(nn.Module):
    def __init__(self, num_classes, in_channels=3, inter_channels=64, kernel_size=3, dropout_rate=0.1, use_weight_norm=True):
        super(Block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, inter_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        self.activ1 = nn.ReLU(inplace=True)
        if use_weight_norm:
            self.conv1 = weight_norm(self.conv1)
        self.dropout1 = nn.Dropout2d(dropout_rate)
        self.conv2 = nn.Conv2d(inter_channels, inter_channels * 2, kernel_size=kernel_size, padding=kernel_size // 2)
        self.activ2 = nn.ReLU(inplace=True)
        if use_weight_norm:
            self.conv2 = weight_norm(self.conv2)
        self.dropout2 = nn.Dropout2d(dropout_rate)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = Rearrange('b c h w -> b (c h w)')
        self.classifier = nn.Linear(inter_channels * 2, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.activ1(x)
        x = self.dropout1(x)
        x = self.conv2(x)
        x = self.activ2(x)
        x = self.dropout2(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.classifier(x)
        return x