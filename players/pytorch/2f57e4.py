import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
from torch.nn.utils import weight_norm

class Block(nn.Module):
    def __init__(self, num_classes, in_channels=3, hidden_layers=(64, 128, 256, 512), kernel_size=3, activation=nn.ReLU, dropout=0.5):
        super(Block, self).__init__()
        conv_blocks = [
            nn.Sequential(
                weight_norm(nn.Conv2d(
                    in_channels if i == 0 else hidden_layers[i-1],
                    hidden_layers[i],
                    kernel_size,
                    padding=kernel_size // 2
                )),
                activation(),
                nn.MaxPool2d(kernel_size=2),
                nn.Dropout(dropout) if dropout > 0 else nn.Identity()
            )
            for i in range(len(hidden_layers))
        ]

        self.feature_extractor = nn.Sequential(*conv_blocks)
        self.classifier = nn.Sequential(
            Rearrange('b c h w -> b (c h w)'),
            nn.Linear(hidden_layers[-1], num_classes)
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        x = nn.AdaptiveAvgPool2d(1)(x)
        x = self.classifier(x)
        return x