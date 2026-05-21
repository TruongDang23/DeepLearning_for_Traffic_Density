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
        self.norm = nn.LayerNorm(num_channels, eps=1e-6)

    def forward(self, x):
        # x: (B, C, H, W) -> (B, H, W, C)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        return x.permute(0, 3, 1, 2)


def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)

    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    binary_tensor = torch.floor(random_tensor)

    output = x.div(keep_prob) * binary_tensor
    return output


class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob
    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)
    

class Stem(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.norm = LayerNorm2d(out_channels)
    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        return x
    

class DownSampling(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm = LayerNorm2d(dim)
        self.conv = nn.Conv2d(dim, dim*2, kernel_size=2, stride=2)
    def forward(self, x):
        x = self.norm(x)
        x = self.conv(x)
        return x

# =========================
# ConvNeXt Block (simplified)
# =========================

class ConvNeXtBlock(nn.Module):
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 7, padding=3, groups=dim)
        self.norm = LayerNorm2d(dim)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)

        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        identity = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = x.permute(0, 2, 3, 1)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)
        return identity + self.drop_path(x)


# =========================
# ConvNeXt Frontend (1/8)
# =========================

# Input: 480 x 640 x 3
class ConvNeXtFrontend(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = Stem(3, 96) # 240 x 320

        self.stage1 = nn.ModuleList()

        for _ in range(3):
            self.stage1.append(ConvNeXtBlock(96))

        self.downsample1 = DownSampling(96) #120 x 160

        self.stage2 = nn.ModuleList()

        for _ in range(3):
            self.stage2.append(ConvNeXtBlock(192))
        
        self.downsample2 = DownSampling(192) # 60 x 80

        self.conv = nn.Conv2d(384, 192, kernel_size=1, stride=1)

    def forward(self, x):
        x = self.stem(x)
        for block in self.stage1:
            x = block(x)
        x = self.downsample1(x)
        for block in self.stage2:
            x = block(x)
        x = self.downsample2(x)
        x = self.conv(x)
        return x  

