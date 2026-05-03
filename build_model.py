import torch
import torch.nn as nn
from torchsummary import summary
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

# Select device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = SwinLayer(dim=dim, 
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
                  fused_window_process=fused_window_process).to(device)

print(model)
summary(model, input_size=(192, 60, 80))