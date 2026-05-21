import torch.nn as nn
import torch
from torchvision import models
from utils import save_net,load_net

class Vgg16(nn.Module):
    def __init__(self, pretrained=True):
            super().__init__()

            vgg = models.vgg16(pretrained=pretrained)

            # Extract feature layers
            features = list(vgg.features.children())

            # VGG16 structure indices:
            # pool1 @ 4
            # pool2 @ 9
            # pool3 @ 16  <-- we stop here

            self.backbone = nn.Sequential(*features[:17])  # up to pool3

    def forward(self, x):
        return self.backbone(x)
        return x               