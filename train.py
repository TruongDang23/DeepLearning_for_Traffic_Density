import sys
import os
import numpy as np
import numpy as np
import argparse
import json
import cv2
import time

import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import datasets, transforms

import dataset
from utils import save_checkpoint
from build_model import CrowdModel

# Global variables
dataset_path = "/mnt/d/common/datasets/TRANCOS_v3"
test_set = "image_sets/test.txt"
train_val_set = "image_sets/trainval.txt"
density_map_set = "density_gt"

# Get device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Argument parser
parser = argparse.ArgumentParser(description='PyTorch Traffic Crowd Estimation Net')
parser.add_argument('--pre', '-p', metavar='PRETRAINED', default=None, type=str,
                    help='path to the pretrained model')
parser.add_argument('--task', '-t', metavar='TASK', type=str, default="may03",
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

def train(train_list, model, criterion, optimizer, epoch):
    losses = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    
    train_loader = torch.utils.data.DataLoader(
        dataset.listDataset(train_list,
                       shuffle=True,
                       transform=transforms.Compose([
                       transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406],
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
        
        loss = criterion(output, target)
        
        losses.update(loss.item(), img.size(0))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()    
        
        batch_time.update(time.time() - end)
        end = time.time()
        
        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  .format(
                   epoch, i, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses))
    
def validate(val_list, model, criterion):
    print ('begin test')
    test_loader = torch.utils.data.DataLoader(
    dataset.listDataset(val_list,
                   shuffle=False,
                   transform=transforms.Compose([
                       transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                std=[0.229, 0.224, 0.225]),
                   ]),  train=False),
    batch_size=args.batch_size)    
    
    model.eval()
    
    mae = 0
    
    for i,(img, target) in enumerate(test_loader):
        img = img.to(device )
        img = Variable(img)
        output = model(img)
        
        mae += abs(output.data.sum()-target.sum().type(torch.FloatTensor).to(device))
    mae = mae/len(test_loader)    
    print(' * MAE {mae:.3f} '
              .format(mae=mae))

    return mae 

def main():
    global args, best_predict
    best_predict = 1e6

    args = parser.parse_args()
    args.original_lr = 1e-7
    args.lr = 1e-7
    args.batch_size    = 1
    args.momentum      = 0.95
    args.decay         = 5*1e-4
    args.start_epoch   = 0
    args.epochs = 400
    args.steps         = [-1,1,100,150]
    args.scales        = [1,1,1,1]
    args.workers = 4
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
    criterion = nn.MSELoss(size_average=False).to(device)
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.decay)
    
    # Loading epoch and train
    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch)

        train(train_list, model, criterion, optimizer, epoch)
        current_predict = validate(val_list, model, criterion)

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