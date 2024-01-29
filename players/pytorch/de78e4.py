import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
from torch.nn.utils import weight_norm

class Block(nn.Module):
    def __init__(self, num_classes, in_channels=3, init_dim=128, depth=4, kernel_size=3, drop_prob=0.05):
        super(Block, self).__init__()
        self.layers = nn.Sequential()
        
        current_dim = in_channels
        for i in range(depth):
            self.layers.add_module(f"weight_norm_conv2d_{i}", weight_norm(nn.Conv2d(current_dim, init_dim * 2**i, kernel_size, padding=kernel_size//2)))
            self.layers.add_module(f"relu_{i}", nn.ReLU(inplace=True))
            if i < depth - 1:  # No pooling on the last layer
                self.layers.add_module(f"pool_{i}", nn.MaxPool2d(2))
            current_dim = init_dim * 2**i
        
        self.gap = nn.AdaptiveAvgPool2d(1)  # Global average pooling
        self.rearrange = Rearrange('b c 1 1 -> b c')
        self.dropout = nn.Dropout(drop_prob)
        self.classifier = nn.Linear(current_dim, num_classes)

    def forward(self, x):
        x = self.layers(x)
        x = self.gap(x)
        x = self.rearrange(x)
        x = self.dropout(x)
        x = self.classifier(x)
        return x