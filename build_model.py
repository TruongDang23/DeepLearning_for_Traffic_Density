import torch
import torch.nn as nn

import torch.nn.functional as F
from torchsummary import summary

from model.backbone import ConvNeXtFrontend
from model.dilated_block import DilatedBranch
from model.swin_block import BasicLayer as SwinLayer

# =========================
# SwinLayer Configure
# =========================
dim = 192 # ConvNext output dimension
input_resolution = (60, 80) # ConvNext output resolution
depth = 2 # Number of swin blocks in each layer
num_heads = 3 # Number of attention heads
window_size = 5 # Window size
mlp_ratio=4. # MLP hidden dimension ratio to embedding dimension
qkv_bias=True
qk_scale=None
drop=0.
attn_drop=0.
drop_path=0.
norm_layer=nn.LayerNorm
downsample=None
use_checkpoint=False
fused_window_process=False
# =========================

swin_block = SwinLayer(dim=dim, 
                  input_resolution=input_resolution, 
                  depth=depth, 
                  num_heads=num_heads, 
                  window_size=window_size, 
                  mlp_ratio=mlp_ratio, 
                  qkv_bias=qkv_bias, 
                  qk_scale=qk_scale, 
                  drop=drop, 
                  attn_drop=attn_drop, 
                  drop_path=drop_path, 
                  norm_layer=norm_layer, 
                  downsample=downsample, 
                  use_checkpoint=use_checkpoint, 
                  fused_window_process=fused_window_process)

# =========================
# Fusion + Decoder
# =========================

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(192, 128, 3, padding=1)
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
        self.swin_branch = swin_block # Input H: 60, W: 80, C: 192; Output H: 60, W: 80, C: 192

        self.fuse = nn.Conv2d(240, 192, 1) #in_C: 240, out_C: 192
        self.decoder = Decoder()

    def forward(self, x):
        B, C, H, W = x.shape
        # the input x should be resize to 480x640 before feeding into the model.
        feat = self.backbone(x) # feat shape: [B, 192, 60, 80]

        conv_out = self.conv_branch(feat)

        feat_swin = feat.view(feat.size(0), feat.size(1), -1) # [B, 192, 60*80]
        feat_swin = torch.permute(feat_swin, (0, 2, 1)).contiguous() # [B, 60*80, 192]
        swin_out = self.swin_branch(feat_swin)
        swin_out = torch.permute(swin_out, (0, 2, 1)).contiguous() # [B, 192, 60*80]
        swin_out = swin_out.view(swin_out.size(0), swin_out.size(1), H // 8, W // 8)

        fused = torch.cat([conv_out, swin_out], dim=1)

        fused = self.fuse(fused)

        out = self.decoder(fused)
        return out


# =========================
# Test Model Summary
# =========================

# # Select device
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # First dry test -----------------------------------
# model = CrowdModel().to(device)
# x = torch.randn(1, 3, 480, 640)
# density_map = model(x)
# print(density_map.shape)

# # Test with image -----------------------------------
# import cv2
# import numpy as np
# import copy
# # Preprocess the image
# img_path = "/mnt/d/common/datasets/TRANCOS_v3/images/image-1-000001.jpg"
# img = cv2.imread(img_path)  # shape: (H, W, C), BGR, uint8
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# ori_img = copy.deepcopy(img)
# img = torch.from_numpy(img)        # (H, W, C), uint8
# img = img.float() / 255.0          # → float32 in [0, 1]
# img = img.permute(2, 0, 1)         # (C, H, W)
# img = img.unsqueeze(0)             # (1, C, H, W)

# # Inference
# model.eval()
# with torch.no_grad():
#     density_map = model(img.to(device))
# # assume output shape: (1, 1, H, W) or (1, H, W)
# density = density_map.squeeze().detach().cpu().numpy()  # (H, W)
# # normalize to 0–255
# density_norm = cv2.normalize(density, None, 0, 255, cv2.NORM_MINMAX)
# # convert to uint8
# density_uint8 = density_norm.astype(np.uint8)
# # apply blue colormap
# heatmap = cv2.applyColorMap(density_uint8, cv2.COLORMAP_JET)
# # show
# combined = cv2.hconcat([ori_img, heatmap])
# cv2.imshow("density", combined)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
