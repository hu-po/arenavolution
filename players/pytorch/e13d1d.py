import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
from torch.nn.utils import weight_norm

class Block(nn.Module):
    def __init__(self, num_classes, in_channels=3, internal_channels=64, kernel_size=3, activation_fn=nn.ReLU, use_weight_norm=True, dropout=0.2):
        super(Block, self).__init__()

        # Define the convolutional block with optional weight normalization
        conv_block = [
            nn.Conv2d(in_channels, internal_channels, kernel_size, padding=kernel_size // 2),
            activation_fn(),
            nn.Conv2d(internal_channels, internal_channels, kernel_size, padding=kernel_size // 2),
            activation_fn()
        ]
        if use_weight_norm:
            conv_block = [weight_norm(layer) if isinstance(layer, nn.Conv2d) else layer for layer in conv_block]
        
        # Pooling layer
        pooling = nn.MaxPool2d(2)

        # Dropout layer
        dropout_layer = nn.Dropout(dropout)
        
        # Combine layers into a block
        self.block = nn.Sequential(*conv_block, pooling, dropout_layer)

        # Adaptive pooling and classifier
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            Rearrange('b c h w -> b (c h w)'),
            nn.Linear(internal_channels, num_classes)
        )
    
    def forward(self, x):
        x = self.block(x)
        x = self.global_pool(x)
        x = self.classifier(x)
        return x