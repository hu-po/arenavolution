import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
from torch.nn.utils import weight_norm

class Block(nn.Module):
    def __init__(self, num_classes, in_channels=3, img_size=224, hidden_dims=None, drop_prob=0.1):
        super(Block, self).__init__()
        if hidden_dims is None:
            hidden_dims = [256, 512]
        
        self.rearrange = Rearrange('b c h w -> b (c h w)')
        layers = [
            weight_norm(nn.Linear(in_channels * img_size * img_size, hidden_dims[0])), 
            nn.ReLU(),
            nn.Dropout(drop_prob)
        ]
        
        for i in range(1, len(hidden_dims)):
            layers.extend([
                weight_norm(nn.Linear(hidden_dims[i-1], hidden_dims[i])),
                nn.ReLU(),
                nn.Dropout(drop_prob)
            ])
        
        self.layers = nn.Sequential(*layers)
        self.classifier = nn.Linear(hidden_dims[-1], num_classes)

    def forward(self, x):
        x = self.rearrange(x)
        x = self.layers(x)
        x = self.classifier(x)
        return x