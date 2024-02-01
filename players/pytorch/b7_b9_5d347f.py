import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
from torch.nn.utils import weight_norm

class Block(nn.Module):
    def __init__(self, num_classes, in_channels=3, hidden_channels=[64, 128], kernel_size=3, expansion_factor=2):
        super(Block, self).__init__()
        self.conv_layers = nn.Sequential()
        for idx, hidden_channel in enumerate(hidden_channels):
            if idx > 0:
                in_channels = hidden_channels[idx - 1]
            conv = nn.Conv2d(in_channels, hidden_channel, kernel_size, padding=kernel_size // 2, bias=False)
            conv = weight_norm(conv)
            self.conv_layers.add_module(f"conv{idx}", conv)
            self.conv_layers.add_module(f"relu{idx}", nn.ReLU())
            self.conv_layers.add_module(f"pool{idx}", nn.MaxPool2d(2))

        self.final_conv_channel = hidden_channels[-1] * expansion_factor
        self.final_conv = weight_norm(nn.Conv2d(hidden_channels[-1], self.final_conv_channel, 1))
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = Rearrange('b c h w -> b (c h w)')
        self.classifier = nn.Linear(self.final_conv_channel, num_classes)

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.final_conv(x)
        x = self.avg_pool(x)
        x = self.flatten(x)
        x = self.classifier(x)
        return x