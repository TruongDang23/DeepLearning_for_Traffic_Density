import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary

# =========================
# Swin Components
# =========================

def window_partition(x, window_size):
    B, C, H, W = x.shape
    x = x.view(B, C, H // window_size, window_size, W // window_size, window_size)
    x = x.permute(0, 2, 4, 3, 5, 1).contiguous()
    return x.view(-1, window_size * window_size, C)


def window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size,
                     window_size, window_size, -1)
    x = x.permute(0, 5, 1, 3, 2, 4).contiguous()
    return x.view(B, -1, H, W)


class WindowAttention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        return self.proj(x)


class SwinBlock(nn.Module):
    def __init__(self, dim=192, num_heads=4, window_size=5, shift_size=2):
        super().__init__()
        self.window_size = window_size
        self.shift_size = shift_size

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, num_heads)

        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )

    def forward(self, x):
        B, C, H, W = x.shape

        # shift
        shifted = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(2, 3))

        # partition
        windows = window_partition(shifted, self.window_size)
        windows = self.norm1(windows)

        attn_windows = self.attn(windows)

        # reverse
        x = window_reverse(attn_windows, self.window_size, H, W)

        # reverse shift
        x = torch.roll(x, shifts=(self.shift_size, self.shift_size), dims=(2, 3))

        # MLP
        x_flat = x.permute(0, 2, 3, 1)
        x_flat = x_flat + self.mlp(self.norm2(x_flat))
        x = x_flat.permute(0, 3, 1, 2)

        return x


class SwinBranch(nn.Module):
    def __init__(self):
        super().__init__()
        self.reduce = nn.Conv2d(256, 192, 1)
        self.block = SwinBlock()
        self.expand = nn.Conv2d(192, 256, 1)

    def forward(self, x):
        x = self.reduce(x)
        x = self.block(x)
        x = self.expand(x)
        return x

