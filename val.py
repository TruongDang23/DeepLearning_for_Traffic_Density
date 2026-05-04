import sys
import os
import numpy as np
import numpy as np
import argparse
import json
import cv2
import time
import h5py
import datetime
import PIL.Image as Image
from matplotlib import cm as CM
from tqdm import tqdm
from pytorch_msssim import ssim, ms_ssim
import torch.nn.functional as F

import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import datasets, transforms
from matplotlib import cm as CM
from matplotlib import pyplot as plt

import dataset
from utils import save_checkpoint
from build_model import CrowdModel

# Global variables
CHECKPOINT = 'checkpoints/may_04_dl_density_best_model.pth.tar'
VIZ = True
BATCH_SIZE = 1
SUBSET = 100
if VIZ is True:
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_folder = f"output/output_viz_{timestamp}"
    os.makedirs(output_folder, exist_ok=True)

#dataset_path = "/mnt/d/common/datasets/TRANCOS_v3"
dataset_path = "/mnt/d/00_master_of_science/linux_workspace/common/datasets/TRANCOS_v3"
test_set = "image_sets/test.txt"
train_val_set = "image_sets/trainval.txt"
density_map_set = "density_gt"

# Get device
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = 'cpu'

def get_image_set(path_set: str):
    img_paths = []
    if ".txt" in path_set:
        with open(path_set, "r") as f:
            count = 0
            for line in f:
                line = line.strip()   # remove newline / whitespace
                img_path = os.path.join(dataset_path, "images", line)
                if os.path.exists(img_path):
                    img_paths.append(img_path)
                    count += 1
                else:
                    print(f">>> Warning: {img_path} does not exist.")
    else:
        print(f">>> Warning: {path_set} is not a valid .txt file.")
        sys.exit(1)
    return img_paths, count

def psnr(pred, target, max_val=1.0):
    mse = F.mse_loss(pred, target) + 1e-8
    return 10 * torch.log10(max_val**2 / mse)

def validate(val_list, model):
    print ('Begin test')
    test_loader = torch.utils.data.DataLoader(
    dataset.listDataset(val_list,
                   shuffle=False,
                   transform=transforms.Compose([
                       transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                std=[0.229, 0.224, 0.225]),
                   ]),  train=False),
    batch_size=1)    
    
    model.eval()
    
    mae = 0
    pnsr_avg = 0
    ssim_avg = 0
    
    for i,(img, target) in enumerate(test_loader):
        img = img.to(device)
        img = Variable(img)
        output = model(img)
        predict_num = output.data.sum()
        target_num = target.sum()
        
        # MAE
        mae += abs(output.data.sum()-target.sum().type(torch.FloatTensor).to(device))

        # Avoid out-of-range
        #pred = torch.clamp(pred, 0, 1)

        # SSIM (hoặc MS-SSIM)
        ssim_val += ms_ssim(output, target, data_range=1.0)

        # PSNR
        psnr_val += psnr(output, target)

        if VIZ is True:
            ori_img = Image.open(val_list[i])
            # Print test image
            # Plot overlay
            plt.figure(figsize=(8, 8))
            plt.imshow(ori_img)
            plt.imshow(output.detach().cpu().numpy().squeeze(0).squeeze(0), cmap=CM.jet, alpha=0.6)  # overlay
            plt.text(10, 80, f'GT  : {int(target_num)}', color='red', fontsize=12)
            plt.text(10, 110, f'PRED: {int(predict_num)}', color='red', fontsize=12)
            plt.axis('off')

            # Save
            img_name = os.path.basename(val_list[i])
            plt.savefig(f"{output_folder}/{img_name}".replace('.jpg', '_overlay.jpg'), bbox_inches='tight', pad_inches=0)
            plt.close()

            # Plot density only
            plt.figure(figsize=(8, 8))
            plt.imshow(output.detach().cpu().numpy().squeeze(0).squeeze(0), cmap=CM.jet) 
            plt.text(10, 80, f'GT  : {int(target_num)}', color='red', fontsize=12)
            plt.text(10, 110, f'PRED: {int(predict_num)}', color='red', fontsize=12)
            plt.axis('off')

            # Save
            plt.savefig(f"{output_folder}/{img_name}".replace('.jpg', '_density_heat.jpg'), bbox_inches='tight', pad_inches=0)
            plt.close()

    mae = mae/len(test_loader)    
    print(' * MAE {mae:.3f} '
              .format(mae=mae))

    return mae 

# Build model
model = CrowdModel().to(device)

# Load checkpoint
checkpoint = torch.load(CHECKPOINT)

# Load state dict
model.load_state_dict(checkpoint['state_dict'])

# Get test set
test_set_path = os.path.join(dataset_path, test_set)
val_list, val_count = get_image_set(test_set_path)
print(f">>> Validation set: {val_count} images found.")

# Validate model
mae = validate(val_list, model)

