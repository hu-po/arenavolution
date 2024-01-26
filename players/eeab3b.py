import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
from torch.nn.utils import weight_norm

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        
    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # binarize
        output = x.div(keep_prob) * random_tensor
        return output

class Block(nn.Module):
    def __init__(self, num_classes, hidden_dim=256, depth=3, drop_path_rate=0.05):
        super(Block, self).__init__()
        self.flatten = Rearrange('b c h w -> b (c h w)')
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        
        layers = []
        for i in range(depth):
            layers.append(nn.Sequential(
                weight_norm(nn.Linear(hidden_dim if i > 0 else 224*224*3, hidden_dim)),
                nn.ReLU(),
                nn.Dropout(0.1),
                DropPath(dpr[i])
            ))
        
        self.blocks = nn.Sequential(*layers)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.flatten(x)
        x = self.blocks(x)
        x = self.classifier(x)
        return x