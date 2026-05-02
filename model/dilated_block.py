import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary

class DilatedBranch(nn.Module):
    def __init__(self, dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, 3, padding=4, dilation=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, 3, padding=6, dilation=6),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)
