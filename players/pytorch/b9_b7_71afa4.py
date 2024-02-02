import torch
import torch.nn as nn

class Block(nn.Module):
    def __init__(self, num_classes, in_channels=3, hidden_dims=[64, 128, 256, 512], kernel_sizes=[3, 3, 3, 3], expansion_factor=4, dropout=0.3):
        super(Block, self).__init__()
        layers = []
        for idx, (hidden_dim, kernel_size) in enumerate(zip(hidden_dims, kernel_sizes)):
            out_channels = hidden_dim * expansion_factor if idx == len(hidden_dims) - 1 else hidden_dim
            conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2, bias=False)
            conv = nn.utils.weight_norm(conv)
            layers.append(conv)
            layers.append(nn.ReLU(inplace=True))
            if idx < len(hidden_dims) - 1:
                layers.append(nn.MaxPool2d(2))
            in_channels = out_channels

        self.features = nn.Sequential(*layers)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dims[-1] * expansion_factor, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.avg_pool(x)
        x = self.dropout(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x