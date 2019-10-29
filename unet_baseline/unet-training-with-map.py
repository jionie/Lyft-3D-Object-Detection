from datetime import datetime
from functools import partial
import glob
from multiprocessing import Pool
import argparse

# Disable multiprocesing for numpy/opencv. We already multiprocess ourselves, this would mean every subprocess produces
# even more threads which would lead to a lot of context switching, slowing things down a lot.
import os
os.environ["OMP_NUM_THREADS"] = "1"

import matplotlib.pyplot as plt
import random
import pandas as pd
import cv2
from PIL import Image
import numpy as np
from tqdm import tqdm, tqdm_notebook
import scipy
import scipy.ndimage
import scipy.special
from scipy.spatial.transform import Rotation as R

import torch
from torch.utils.data import TensorDataset, DataLoader,Dataset
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.optim.optimizer import Optimizer
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.utils.data.sampler import SubsetRandomSampler
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau, CosineAnnealingLR, _LRScheduler
import torch.nn.utils.weight_norm as weightNorm
import torch.nn.init as init
from torch.nn.parallel.data_parallel import data_parallel

from models.model import *
import torchvision.models as models

from utils.transform import *

from tensorboardX import SummaryWriter
from apex import amp
from ranger import *
import learning_schedules_fastai as lsf
from fastai_optim import OptimWrapper

import albumentations
from albumentations import torch as AT

parser = argparse.ArgumentParser(description="arg parser")
parser.add_argument('--model', type=str, default='seresnext101', required=False, help='specify the backbone model')
parser.add_argument('--optimizer', type=str, default='Ranger', required=False, help='specify the optimizer')
parser.add_argument("--lr_scheduler", type=str, default='adamonecycle', required=False, help="specify the lr scheduler")
parser.add_argument("--batch_size", type=int, default=16, required=False, help="specify the batch size for training")
parser.add_argument("--valid_batch_size", type=int, default=64, required=False, help="specify the batch size for validating")
parser.add_argument("--num_epoch", type=int, default=50, required=False, help="specify the total epoch")
parser.add_argument("--accumulation_steps", type=int, default=4, required=False, help="specify the accumulation steps")
parser.add_argument("--start_epoch", type=int, default=0, required=False, help="specify the start epoch for continue training")
parser.add_argument("--train_data_folder", type=str, default="/media/jionie/my_disk/Kaggle/Lyft/input/3d-object-detection-for-autonomous-vehicles/train_root/bev_data_with_map/", \
    required=False, help="specify the folder for training data")
parser.add_argument("--checkpoint_folder", type=str, default="/media/jionie/my_disk/Kaggle/Lyft/model/unet", \
    required=False, help="specify the folder for checkpoint")
parser.add_argument('--load_pretrain', action='store_true', default=True, help='whether to load pretrain model')

############################################################################## seed all
SEED = 42
def seed_everything(seed=SEED):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
seed_everything(SEED)



############################################################################## define trainsformation
SIZE = 336


def transform_train(image, mask):
    if random.random() < 0.5:
        image = albumentations.RandomRotate90(p=1)(image=image)['image']
        mask = albumentations.RandomRotate90(p=1)(image=mask)['image']

    if random.random() < 0.5:
        image = albumentations.Transpose(p=1)(image=image)['image']
        mask = albumentations.Transpose(p=1)(image=mask)['image']

    if random.random() < 0.5:
        image = albumentations.VerticalFlip(p=1)(image=image)['image']
        mask = albumentations.VerticalFlip(p=1)(image=mask)['image']

    if random.random() < 0.5:
        image = albumentations.HorizontalFlip(p=1)(image=image)['image']
        mask = albumentations.HorizontalFlip(p=1)(image=mask)['image']

    # if random.random() < 0.5:
    #     image = albumentations.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.15, rotate_limit=45, p=1)(image=image)['image']
    #     mask = albumentations.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.15, rotate_limit=45, p=1)(image=mask)['image']

    # if random.random() < 0.5:
    #     image = albumentations.RandomBrightness(0.1)(image=image)['image']
    #     image = albumentations.RandomContrast(0.1)(image=image)['image']
    #     image = albumentations.Blur(blur_limit=3)(image=image)['image']

    # if random.random() < 0.5:
    #     image = albumentations.Cutout(num_holes=1, max_h_size=32, max_w_size=32, p=1)(image)
    #     mask = albumentations.Cutout(num_holes=1, max_h_size=32, max_w_size=32, p=1)(mask)

    return image, mask

def transform_valid(image, mask):
    # if random.random() < 0.5:
    #     image = albumentations.RandomRotate90(p=1)(image=image)['image']
    #     mask = albumentations.RandomRotate90(p=1)(image=mask)['image']

    # if random.random() < 0.5:
    #     image = albumentations.Transpose(p=1)(image=image)['image']
    #     mask = albumentations.Transpose(p=1)(image=mask)['image']

    # if random.random() < 0.5:
    #     image = albumentations.VerticalFlip(p=1)(image=image)['image']
    #     mask = albumentations.VerticalFlip(p=1)(image=mask)['image']

    if random.random() < 0.5:
        image = albumentations.HorizontalFlip(p=1)(image=image)['image']
        mask = albumentations.HorizontalFlip(p=1)(image=mask)['image']

    return image, mask

def transform_test(image):
    
    image_hard = image.copy()
    image_simple = image.copy()

    if random.random() < 0.5:
        image_hard = albumentations.RandomBrightness(0.1)(image=image_hard)['image']
        image_hard = albumentations.RandomContrast(0.1)(image=image_hard)['image']
        image_hard = albumentations.Blur(blur_limit=3)(image=image_hard)['image']

    return image_simple, image_hard



############################################################################## define bev dataset
class BEVImageDataset(torch.utils.data.Dataset):
    def __init__(self, input_filepaths=None, target_filepaths=None, type="train", img_size=336, map_filepaths=None):
        self.input_filepaths = input_filepaths
        self.target_filepaths = target_filepaths
        self.type = type
        self.map_filepaths = map_filepaths
        self.img_size = img_size
        
        if map_filepaths is not None:
            assert len(input_filepaths) == len(map_filepaths)
        
        if (self.type != "test"):
            assert len(input_filepaths) == len(target_filepaths)

    def __len__(self):
        return len(self.input_filepaths)

    def __getitem__(self, idx):
        input_filepath = self.input_filepaths[idx]
        
        sample_token = input_filepath.split("/")[-1].replace("_input.png","")
        
        im = cv2.imread(input_filepath, cv2.IMREAD_UNCHANGED)
        
        if self.map_filepaths:
            map_filepath = self.map_filepaths[idx]
            map_im = cv2.imread(map_filepath, cv2.IMREAD_UNCHANGED)
            im = np.concatenate((im, map_im), axis=2)

        if (self.target_filepaths):
            target_filepath = self.target_filepaths[idx]
            target = cv2.imread(target_filepath, cv2.IMREAD_UNCHANGED)
            target = target.astype(np.int64)
        else:
            target = None

        if (self.type == "train"):
            im, target = transform_train(im, target)
        elif (self.type == "valid"):
            im, target = transform_valid(im, target)
        else:
            im, _ = transform_test(im) # im_simple, im_hard
            
        im = im.astype(np.float32)/255
        
        im = torch.from_numpy(im.transpose(2,0,1))

        if (self.type != "test"):
            target = torch.from_numpy(target)
            return im, target, sample_token
        else:
            return im, sample_token

def children(m: nn.Module):
    return list(m.children())

def num_children(m: nn.Module):
    return len(children(m))

def unet_training(model_name,
                  optimizer_name,
                  lr_scheduler_name,
                  batch_size,
                  valid_batch_size,
                  num_epoch,
                  start_epoch,
                  accumulation_steps,
                  train_data_folder, 
                  checkpoint_folder,
                  load_pretrain
                  ):
    ############################################################################## train test splitting 0.8 / 0.2
    input_filepaths = sorted(glob.glob(os.path.join(train_data_folder, "*_input.png")))
    target_filepaths = sorted(glob.glob(os.path.join(train_data_folder, "*_target.png")))
    map_filepaths = sorted(glob.glob(os.path.join(train_data_folder, "*_map.png")))

    train_input_filepaths = input_filepaths[:int(0.8*len(input_filepaths))]
    train_target_filepaths = target_filepaths[:int(0.8*len(target_filepaths))]
    train_map_filepaths = map_filepaths[:int(0.8*len(map_filepaths))]
    
    valid_input_filepaths = input_filepaths[int(0.8*len(input_filepaths)):]
    valid_target_filepaths = target_filepaths[int(0.8*len(target_filepaths)):]
    valid_map_filepaths = map_filepaths[int(0.8*len(map_filepaths)):]

    train_dataset = BEVImageDataset(input_filepaths=train_input_filepaths, target_filepaths=train_target_filepaths, \
        type="train", img_size=SIZE, map_filepaths=train_map_filepaths)
    valid_dataset = BEVImageDataset(input_filepaths=valid_input_filepaths, target_filepaths=valid_target_filepaths, \
        type="valid", img_size=SIZE, map_filepaths=valid_map_filepaths)



    ############################################################################## define unet model with backbone
    def get_unet_model(model_name="efficientnet-b3", IN_CHANNEL=6, NUM_CLASSES=2, SIZE=336):
        model = model_iMet(model_name, IN_CHANNEL, NUM_CLASSES, SIZE)
        
        # Optional, for multi GPU training and inference
        # model = nn.DataParallel(model)
        return model

    # We weigh the loss for the 0 class lower to account for (some of) the big class imbalance.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    classes = ["car", "motorcycle", "bus", "bicycle", "truck", "pedestrian", "other_vehicle", "animal", "emergency_vehicle"]
    class_weights = torch.from_numpy(np.array([0.2] + [1.0]*len(classes), dtype=np.float32))
    class_weights = class_weights.to(device)



    ############################################################################### training parameters
    train_batch_size = batch_size
    checkpoint_filename = model_name + "_unet_checkpoint.pth"
    checkpoint_filepath = os.path.join(checkpoint_folder, checkpoint_filename)


    ############################################################################### model and optimizer
    model = get_unet_model(model_name=model_name, IN_CHANNEL=6, NUM_CLASSES=len(classes)+1, SIZE=SIZE)
    if (load_pretrain):
        model.load_pretrain(checkpoint_filepath)
    model = model.to(device)

    # optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    lr = 1e-3
    
    if optimizer_name == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    elif optimizer_name == "adamonecycle":
        flatten_model = lambda m: sum(map(flatten_model, m.children()), []) if num_children(m) else [m]
        get_layer_groups = lambda m: [nn.Sequential(*flatten_model(m))]

        optimizer_func = partial(optim.Adam, betas=(0.9, 0.99))
        optimizer = OptimWrapper.create(
            optimizer_func, 3e-3, get_layer_groups(model), wd=1e-4, true_wd=True, bn_wd=True
        )
    elif optimizer_name == "Ranger":
        optimizer = Ranger(filter(lambda p: p.requires_grad, model.parameters()), lr, weight_decay=1e-5)
    else:
        raise NotImplementedError
    
    if lr_scheduler_name == "adamonecycle":
        scheduler = lsf.OneCycle(optimizer, len(train_dataset) * num_epoch, lr, [0.95, 0.85], 10.0, 0.4)
        lr_scheduler_each_iter = True
    elif lr_scheduler_name == "CosineAnealing":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epoch, eta_min=0, last_epoch=-1)
        lr_scheduler_each_iter = False
    else:
        raise NotImplementedError
        
    model, optimizer = amp.initialize(model, optimizer, opt_level="O1")
    ############################################################################### dataloader
    valid_dataloader = torch.utils.data.DataLoader(valid_dataset, valid_batch_size, shuffle=False, num_workers=os.cpu_count()*2)



    ############################################################################### training
    writer = SummaryWriter()
    log_file = open(model_name + "_log.txt", "a+")
    valid_metric_optimal = np.inf

    for epoch in range(1, num_epoch+1):
        
        if (epoch < start_epoch):
            continue
        
        print("Epoch", epoch)
        print("Epoch", epoch, file = log_file)


        seed_everything(SEED+epoch)

        train_dataloader = torch.utils.data.DataLoader(train_dataset, train_batch_size, shuffle=True, num_workers=os.cpu_count()*2)
        eval_step = len(train_dataloader)  
        train_losses = []
        valid_losses = []
        valid_ce_losses = []

        torch.cuda.empty_cache()
        
        if (not lr_scheduler_each_iter):
            scheduler.step(epoch)
            
        optimizer.zero_grad()
        
        for tr_batch_i, (X, target, sample_ids) in enumerate(train_dataloader):
            
            if (lr_scheduler_each_iter):
                scheduler.step(tr_batch_i)

            model.train() 

            X = X.to(device).float()  # [N, 6, H, W]
            target = target.to(device)  # [N, H, W] with class indices (0, 1)
            
            prediction, _ = model(X)  # [N, 2, H, W]
            loss = F.cross_entropy(prediction, target, weight=class_weights)

            target = target.clone().unsqueeze_(1)
            one_hot = torch.cuda.FloatTensor(target.size(0), len(classes)+1, target.size(2), target.size(3)).zero_()
            target_one_hot = one_hot.scatter_(1, target.data, 1)
            loss += model.get_loss(SIZE, prediction, fc=None, labels=target_one_hot)

            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()

            #loss.backward()
        
            if ((tr_batch_i+1)%accumulation_steps==0):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0, norm_type=2)
                optimizer.step()
                optimizer.zero_grad()

                writer.add_scalar('train_loss', loss.item()*accumulation_steps, epoch*len(train_data_folder)*train_batch_size+tr_batch_i*train_batch_size)
            
            train_losses.append(loss.detach().cpu().numpy())

            if (tr_batch_i+1)%eval_step == 0:  

                with torch.no_grad():
                    
                    torch.cuda.empty_cache()

                    for val_batch_i, (X, target, sample_ids) in enumerate(valid_dataloader):

                        model.eval()

                        X = X.to(device).float()  # [N, 3, H, W]
                        target = target.to(device)  # [N, H, W] with class indices (0, 1)
                        prediction, _ = model(X)  # [N, 2, H, W]

                        ce_loss = F.cross_entropy(prediction, target, weight=class_weights).detach().cpu().numpy()

                        target = target.unsqueeze_(1)
                        one_hot = torch.cuda.FloatTensor(target.size(0), len(classes)+1, target.size(2), target.size(3)).zero_()
                        target_one_hot = one_hot.scatter_(1, target.data, 1)
                        loss = model.get_loss(SIZE, prediction, fc=None, labels=target_one_hot)

                        writer.add_scalar('val_loss', loss, epoch*len(valid_dataloader)*valid_batch_size+val_batch_i*valid_batch_size)

                        valid_losses.append(loss.detach().cpu().numpy())
                        valid_ce_losses.append(ce_loss)
            
        print("Train Loss:", np.mean(train_losses), "Valid Loss:", np.mean(valid_losses), "Valid CE Loss:", np.mean(valid_ce_losses))
        print("Train Loss:", np.mean(train_losses), "Valid Loss:", np.mean(valid_losses), "Valid CE Loss:", np.mean(valid_ce_losses), file = log_file)

        val_metric_epoch = np.mean(valid_losses)

        if (val_metric_epoch <= valid_metric_optimal):

            print('Validation metric improved ({:.6f} --> {:.6f}).  Saving model ...'.format(\
                    valid_metric_optimal, val_metric_epoch))

            print('Validation metric improved ({:.6f} --> {:.6f}).  Saving model ...'.format(\
            valid_metric_optimal, val_metric_epoch), file=log_file)

            valid_metric_optimal = val_metric_epoch
            torch.save(model.state_dict(), checkpoint_filepath)
            
if __name__ == "__main__":
    
    args = parser.parse_args()
    
    unet_training(args.model, args.optimizer, args.lr_scheduler, args.batch_size, args.valid_batch_size, \
                    args.num_epoch, args.start_epoch, args.accumulation_steps, args.train_data_folder, \
                    args.checkpoint_folder, args.load_pretrain)
