import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
from torch.nn.utils import spectral_norm

class Block(nn.Module):
    def __init__(self, num_classes, in_channels=3, img_size=224, hidden_dims=[256, 512], drop_prob=0.05):
        super(Block, self).__init__()
        self.rearrange = Rearrange('b c h w -> b (c h w)')
        layers = [nn.Linear(in_channels * img_size * img_size, hidden_dims[0]), nn.LeakyReLU(0.2), nn.Dropout(drop_prob)]
        for i in range(1, len(hidden_dims)):
            layers.extend([
                spectral_norm(nn.Linear(hidden_dims[i - 1], hidden_dims[i])),
                nn.LeakyReLU(0.2),
                nn.Dropout(drop_prob)
            ])
        layers.append(nn.Linear(hidden_dims[-1], num_classes))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        x = self.rearrange(x)
        x = self.layers(x)
        return x