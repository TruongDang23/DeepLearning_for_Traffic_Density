import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary

# =========================
# Utility Layers
# =========================

class LayerNorm2d(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        self.norm = nn.LayerNorm(num_channels)

    def forward(self, x):
        # x: (B, C, H, W) -> (B, H, W, C)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        return x.permute(0, 3, 1, 2)


# =========================
# ConvNeXt Block (simplified)
# =========================

class ConvNeXtBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 7, padding=3, groups=dim)
        self.norm = LayerNorm2d(dim)
        self.pwconv1 = nn.Conv2d(dim, 4 * dim, 1)
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv2d(4 * dim, dim, 1)

    def forward(self, x):
        identity = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        return x + identity


# =========================
# ConvNeXt Frontend (1/8)
# =========================

class ConvNeXtFrontend(nn.Module):
    def __init__(self):
        super().__init__()

        self.stage1 = nn.Sequential(
            nn.Conv2d(3, 64, 4, stride=2, padding=1),  # 320→160
            ConvNeXtBlock(64)
        )

        self.stage2 = nn.Sequential(
            nn.Conv2d(64, 128, 2, stride=2),  # 160→80
            ConvNeXtBlock(128)
        )

        self.stage3 = nn.Sequential(
            nn.Conv2d(128, 256, 2, stride=2),  # 80→40
            ConvNeXtBlock(256)
        )

    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        return x  # (B, 256, 40, 80)

