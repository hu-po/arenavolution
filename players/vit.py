import torch
from torch import nn
from einops.layers.torch import EinMix as Mix


def patcher(patch_size=16, in_channels=3, num_features=128):
    return Mix(
        "b c_in (h hp) (w wp) -> b (h w) c",
        weight_shape="c_in hp wp c",
        bias_shape="c",
        c=num_features,
        hp=patch_size,
        wp=patch_size,
        c_in=in_channels,
    )


# must be called "Block" but this is a simple ViT
class Block(nn.Module):
    def __init__(
        self,
        image_size=224,
        patch_size=16,
        num_patches=196,
        num_classes=10,
        dim=128,
        depth=12,
        heads=8,
        mlp_dim=256,
    ):
        super(Block, self).__init__()

        self.patcher = patcher(patch_size=patch_size, in_channels=3, num_features=dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, dim))
        self.transformer = nn.Sequential(
            *[
                nn.TransformerEncoderLayer(
                    d_model=dim, nhead=heads, dim_feedforward=mlp_dim
                )
                for _ in range(depth)
            ]
        )

        self.to_cls_token = nn.Identity()
        self.mlp_head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, num_classes))

    def forward(self, x):
        x = self.patcher(x)
        b, n, _ = x.shape
        x += self.pos_embedding[:, :n]
        x = self.transformer(x)
        x = self.to_cls_token(x[:, 0])
        return self.mlp_head(x)
