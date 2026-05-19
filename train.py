import sys
import os
import numpy as np
import numpy as np
import argparse
import json
import cv2
import time
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import datasets, transforms
import torch.nn.functional as F
#from pytorch_msssim import ssim, ms_ssim
from torchmetrics import StructuralSimilarityIndexMeasure

import dataset
from utils import save_checkpoint
from build_model import CrowdModel


# Global variables
dataset_path = "/mnt/d/common/datasets/TRANCOS_v3"
#dataset_path = "/home/nghia/ws/master_project/datasets/TRANCOS_v3"
test_set = "image_sets/test.txt"
train_val_set = "image_sets/trainval.txt"
density_map_set = "density_gt"

# Get device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Argument parser
parser = argparse.ArgumentParser(description='PyTorch Traffic Crowd Estimation Net')
parser.add_argument('--pre', '-p', metavar='PRETRAINED', default=None, type=str,
                    help='path to the pretrained model')
parser.add_argument('--task', '-t', metavar='TASK', type=str, default="may05",
                    help='task id to use.')

# Measuerment
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count 

# To get train list files and test list files
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

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    
    args.lr = args.original_lr
    
    for i in range(len(args.steps)):
        
        scale = args.scales[i] if i < len(args.scales) else 1
        
        
        if epoch >= args.steps[i]:
            args.lr = args.lr * scale
            if epoch == args.steps[i]:
                break
        else:
            break
    for param_group in optimizer.param_groups:
        param_group['lr'] = args.lr

def train(train_list, model, optimizer, epoch):
    losses = AverageMeter()
    mse_loss_meter = AverageMeter()
    mse_meter = AverageMeter()
    mae_meter = AverageMeter()
    grid_meter = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    
    train_loader = torch.utils.data.DataLoader(
        dataset.listDataset(train_list,
                       shuffle=True,
                       transform=transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                        std=[0.229, 0.224, 0.225]),
                   ]), 
                       train=True, 
                       #seen=model.seen,
                       seen=0,
                       batch_size=args.batch_size,
                       num_workers=args.workers),
        batch_size=args.batch_size)
    print('epoch %d, processed %d samples, lr %.10f' % (epoch, epoch * len(train_loader.dataset), args.lr))
    
    model.train()
    end = time.time()
    
    for i,(img, target)in enumerate(train_loader):
        data_time.update(time.time() - end)

        img = img.to(device)
        img = Variable(img)
        output = model(img)
    
        target = target.type(torch.FloatTensor).unsqueeze(0).to(device)
        target = Variable(target)
        
        loss, mse_loss, mse, mae, grid_loss = density_loss(output, target)
        
        losses.update(loss.item(), img.size(0))
        mse_meter.update(mse.item(), img.size(0))
        mae_meter.update(mae.item(), img.size(0))
        mse_loss_meter.update(mse_loss.item(), img.size(0))
        grid_meter.update(grid_loss.item(), img.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()    
        
        batch_time.update(time.time() - end)
        end = time.time()
        
        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  #'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  #'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'LOSS {loss.val:.4f} ({loss.avg:.4f})\t'
                  'MSE_LOSS {mse_loss.val:.4f} ({mse_loss.avg:.4f})\t'
                  'MSE {mse.val:.4f} ({mse.avg:.4f})\t'
                  'MAE {mae.val:.4f} ({mae.avg:.4f})\t'
                  'GRID_L {grid_loss.val:.4f} ({grid_loss.avg:.4f})\t'
                  .format(
                   epoch, i, len(train_loader), 
                   #batch_time=batch_time,
                   #data_time=data_time, 
                   loss=losses, 
                   mse=mse_meter, 
                   mae=mae_meter, 
                   mse_loss=mse_loss_meter,
                   grid_loss = grid_meter))
    
def validate(val_list, model):
    print ('Begin test')
    test_loader = torch.utils.data.DataLoader(
    dataset.listDataset(val_list,
                   shuffle=False,
                   transform=transforms.Compose([
                       transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                std=[0.229, 0.224, 0.225]),
                   ]),  train=False),
    batch_size=args.batch_size)    
    
    model.eval()
    
    loss_val_avg = 0
    mse_loss_val_avg = 0
    mse_val_avg = 0
    mae_val_avg = 0
    gird_loss_val_avg = 0
    
    for i, (img, target) in enumerate(tqdm(test_loader)):
        img = img.to(device)
        img = Variable(img)
        output = model(img)

        target = target.type(torch.FloatTensor).unsqueeze(0).to(device)
        target = Variable(target)
        
        loss, mse_loss, mse, mae, grid_loss = density_loss(output, target)
        
        loss_val_avg += loss
        mse_loss_val_avg += mse_loss
        mse_val_avg += mse
        mae_val_avg += mae
        gird_loss_val_avg += grid_loss
        
    loss_val_avg = loss_val_avg/len(test_loader)
    mse_loss_val_avg = mse_loss_val_avg/len(test_loader)
    mse_val_avg = mse_val_avg/len(test_loader)
    mae_val_avg = mae_val_avg/len(test_loader)
    gird_loss_val_avg = gird_loss_val_avg/len(test_loader)

    print(' * LOSS {loss_val_avg:.3f} \t'
          ' MSE_LOSS {mse_loss_val_avg:.3f} \t'
          ' MSE {mse_val_avg:.3f} \t'
          ' MAE {mae_val_avg:.3f} \t'
          ' GRID_LOSS {gird_loss_val_avg:.3f} \t'
              .format(loss_val_avg=loss_val_avg, 
                      mse_loss_val_avg=mse_loss_val_avg,
                      mse_val_avg=mse_val_avg,
                      mae_val_avg=mae_val_avg,
                      gird_loss_val_avg=gird_loss_val_avg))

    return loss_val_avg, mse_loss_val_avg, mse_val_avg, mae_val_avg, gird_loss_val_avg

# def psnr(pred, target, max_val=1.0):
#     mse = F.mse_loss(pred, target) + 1e-8
#     return 10 * torch.log10(max_val**2 / mse)

def regional_loss(pred, gt, level=1):

    B, C, H, W = pred.shape
    gt = gt.type(torch.FloatTensor).unsqueeze(0).to(device)

    grid = 2 ** level

    pred = pred.view(
        B, C,
        grid, H // grid,
        grid, W // grid
    )

    gt = gt.view(
        B, C,
        grid, H // grid,
        grid, W // grid
    )

    pred_cnt = pred.sum(dim=(3,5))
    gt_cnt   = gt.sum(dim=(3,5))

    loss = abs((pred_cnt - gt_cnt)).sum()

    return loss

def density_loss(pred, target):
    global ssim_loss, mse_loss
    B = pred.shape[0]
    mse_loss = ((pred.sum() - target.sum()) ** 2) / (2*B)
    mae = (pred.sum() - target.sum()).abs() / B
    mse = ((pred.sum() - target.sum()) ** 2) / B

    # Grid loss
    grid_loss = regional_loss(pred, target, level=2)

    total = mse_loss
    return total, mse_loss, mse, mae, grid_loss

def main():
    global args, best_predict
    best_predict = 1e6

    args = parser.parse_args()
    args.original_lr = 1e-6
    args.lr = 1e-6
    args.batch_size    = 1
    args.momentum      = 0.95
    args.decay         = 5*1e-4
    args.start_epoch   = 0
    args.epochs = 200
    args.steps         = [-1,1,100,150]
    args.scales        = [1,1,1,1]
    args.workers = 1
    args.seed = time.time()
    args.print_freq = 30

    # Get train list and test list
    train_val_set_path = os.path.join(dataset_path, train_val_set) 
    train_list, train_count = get_image_set(train_val_set_path)
    print(f">>> Train set: {train_count} images found.")
    test_set_path = os.path.join(dataset_path, test_set)
    val_list, val_count = get_image_set(test_set_path)
    print(f">>> Validation set: {val_count} images found.")

    # Build model
    model = CrowdModel().to(device)

    # Criterion and optimizer
    global ssim_loss, mse_loss
    ssim_loss = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    mse_loss = nn.MSELoss().to(device)
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.decay)
    
    if args.pre:
        if os.path.isfile(args.pre):
            print("=> loading checkpoint '{}'".format(args.pre))
            checkpoint = torch.load(args.pre)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.pre, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.pre))

    # Loading epoch and train
    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch)

        train(train_list, model, optimizer, epoch)
        current_predict = validate(val_list, model)

        is_best = current_predict < best_predict
        best_predict = min(current_predict, best_predict)

        print(' * best MAE {mae:.3f} '
              .format(mae=best_predict))
        
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.pre,
            'state_dict': model.state_dict(),
            'best_prec1': best_predict,
            'optimizer' : optimizer.state_dict(),
        }, is_best, args.task)

if __name__ == '__main__':
    main() 