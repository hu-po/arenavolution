import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
from torch.nn.utils import weight_norm

class Block(nn.Module):
    def __init__(self, num_classes, in_channels=3, hidden_dims=None, kernel_size=3, activation_fn=nn.ReLU, use_weight_norm=True):
        super(Block, self).__init__()

        if hidden_dims is None:
            hidden_dims = [64, 128, 256, 512]

        # Construct the sequential layers for convolutional operations
        conv_layers = []
        for i, hidden_dim in enumerate(hidden_dims):
            conv_layer = nn.Conv2d(in_channels=in_channels if i == 0 else hidden_dims[i-1], out_channels=hidden_dim, kernel_size=kernel_size, padding=kernel_size//2)
            if use_weight_norm:
                conv_layer = weight_norm(conv_layer)

            conv_layers.extend([
                conv_layer,
                activation_fn(),
                nn.MaxPool2d(kernel_size=2)
            ])

        # Add the final classifier layer
        classifier_layers = [
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            Rearrange('b c h w -> b (c h w)'),
            nn.Linear(hidden_dims[-1], num_classes)
        ]

        # Combine convolutional layers and classifier layers
        self.model = nn.Sequential(*conv_layers, *classifier_layers)

    def forward(self, x):
        return self.model(x)