import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary

# =========================
# Dilated Conv Branch
# =========================

# =========================
# Fusion + Decoder
# =========================

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(256, 128, 3, padding=1)
        self.conv2 = nn.Conv2d(128, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 32, 3, padding=1)
        self.out = nn.Conv2d(32, 1, 1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = self.conv1(x)

        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = self.conv2(x)

        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        x = self.conv3(x)

        return self.out(x)


class CrowdModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = ConvNeXtFrontend()
        self.conv_branch = DilatedBranch()
        self.swin_branch = SwinBranch()

        self.fuse = nn.Conv2d(512, 256, 1)
        self.decoder = Decoder()

    def forward(self, x):
        feat = self.backbone(x)

        conv_out = self.conv_branch(feat)
        swin_out = self.swin_branch(feat)

        fused = torch.cat([conv_out, swin_out], dim=1)
        fused = self.fuse(fused)

        out = self.decoder(fused)
        return out


# =========================
# Test
# =========================

device = torch.device("cpu")

model = CrowdModel().to(device)

summary(model, input_size=(16, 3, 320, 640))