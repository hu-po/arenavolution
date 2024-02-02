import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
from torch.nn.utils import weight_norm

class Block(nn.Module):
    def __init__(self, num_classes, in_channels=3, hidden_channels=(128, 256), activation_fn=nn.ReLU, use_weight_norm=True, dropout_rate=0.1):
        super(Block, self).__init__()
        layers = []
        channels = in_channels
        for hidden_channel in hidden_channels:
            conv_layer = nn.Conv2d(channels, hidden_channel, kernel_size=3, stride=1, padding=1)
            if use_weight_norm:
                conv_layer = weight_norm(conv_layer)
            layers += [
                conv_layer,
                activation_fn(),
                nn.MaxPool2d(kernel_size=2),
                nn.Dropout(dropout_rate)
            ]
            channels = hidden_channel
        
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            Rearrange('b c h w -> b (c h w)'),
            nn.Linear(hidden_channels[-1], num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x