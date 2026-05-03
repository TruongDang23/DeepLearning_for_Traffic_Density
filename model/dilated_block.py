import torch
import torch.nn as nn

class DilatedBranch(nn.Module):
    def __init__(self, dim=192):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=2, dilation=2), #192 channels
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, 3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim, 3, padding=2, dilation=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, dim//2, 3, padding=2, dilation=2), #96 channels
            nn.ReLU(inplace=True),
            nn.Conv2d(dim//2, dim//4, 3, padding=2, dilation=2), #48 channels
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)
