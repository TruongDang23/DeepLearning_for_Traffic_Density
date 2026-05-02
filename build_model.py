import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary
from model.backbone import ConvNeXtFrontend
from model.dilated_block import DilatedBranch
from model.swin_block import SwinBranch

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
        self.backbone = ConvNeXtFrontend() # Input H: 480, W: 640, C: 3; Ouput H: 60, W: 80, C: 192
        self.conv_branch = DilatedBranch() # Input H: 60, W: 80, C: 192; Output H: 60, W: 80, C: 48
        self.swin_branch = SwinBranch() # Input H: 60, W: 80, C: 192; Output H: 60, W: 80, C: 192

        self.fuse = nn.Conv2d(240, 120, 1)
        self.decoder = Decoder()

    def forward(self, x):
        # Resize về (H=480, W=640)
        x = F.interpolate(x, size=(480, 640), mode='bilinear', align_corners=False)

        feat = self.backbone(x)

        conv_out = self.conv_branch(feat)
        swin_out = self.swin_branch(feat)

        fused = torch.cat([conv_out, swin_out], dim=1)
        fused = self.fuse(fused)

        out = self.decoder(fused)
        return out


# =========================
# Test Model Summary
# =========================

device = torch.device("cpu")

model = CrowdModel().to(device)

summary(model, input_size=(16, 3, 480, 640))