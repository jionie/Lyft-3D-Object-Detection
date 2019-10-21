from datetime import datetime
from functools import partial
import glob
from multiprocessing import Pool

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

import albumentations
from albumentations import torch as AT



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
        
        im = torch.from_numpy(im.transpose(2,0,1))

        if (self.type != "test"):
            target = torch.from_numpy(target)
            return im, target, sample_token
        else:
            return im, sample_token



############################################################################## train test splitting 0.8 / 0.2
train_data_folder = "/media/jionie/my_disk/Kaggle/Lyft/input/3d-object-detection-for-autonomous-vehicles/bev_train_data/"

input_filepaths = sorted(glob.glob(os.path.join(train_data_folder, "*_input.png")))
target_filepaths = sorted(glob.glob(os.path.join(train_data_folder, "*_target.png")))

train_input_filepaths = input_filepaths[:int(0.8*len(input_filepaths))]
train_target_filepaths = target_filepaths[:int(0.8*len(target_filepaths))]
valid_input_filepaths = input_filepaths[int(0.8*len(input_filepaths)):]
valid_target_filepaths = target_filepaths[int(0.8*len(target_filepaths)):]

train_dataset = BEVImageDataset(input_filepaths=train_input_filepaths, target_filepaths=train_target_filepaths, type="train", img_size=SIZE, map_filepaths=None)
valid_dataset = BEVImageDataset(input_filepaths=valid_input_filepaths, target_filepaths=valid_target_filepaths, type="valid", img_size=SIZE, map_filepaths=None)



############################################################################## define unet model with backbone
def get_unet_model(model_name="efficient-b3", IN_CHANNEL=3, NUM_CLASSES=2, SIZE=336):
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
train_batch_size = 16
valid_batch_size = 64
num_epoch = 100
accumulation_steps = 4
checkpoint_filename = "unet_checkpoint.pth"
checkpoint_filepath = os.path.join("/media/jionie/my_disk/Kaggle/Lyft/model/unet", checkpoint_filename)


############################################################################### model and optimizer
model = get_unet_model(model_name="efficientnet-b3", IN_CHANNEL=3, NUM_CLASSES=len(classes)+1, SIZE=SIZE)
model.load_pretrain(checkpoint_filepath)
model = model.to(device)

# optim = torch.optim.Adam(model.parameters(), lr=1e-3)
lr = 1e-3
optimizer = Ranger(filter(lambda p: p.requires_grad, model.parameters()), lr, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epoch, eta_min=0, last_epoch=-1)
model, optimizer = amp.initialize(model, optimizer, opt_level="O1")



############################################################################### dataloader
valid_dataloader = torch.utils.data.DataLoader(valid_dataset, valid_batch_size, shuffle=False, num_workers=os.cpu_count()*2)



############################################################################### training
writer = SummaryWriter()
log_file = open("log.txt", "w")
valid_metric_optimal = np.inf

for epoch in range(9, num_epoch+1):
    print("Epoch", epoch)
    print("Epoch", epoch, file = log_file)


    seed_everything(SEED+epoch)

    train_dataloader = torch.utils.data.DataLoader(train_dataset, train_batch_size, shuffle=True, num_workers=os.cpu_count()*2)
    eval_step = len(train_dataloader)  
    train_losses = []
    valid_losses = []
    valid_ce_losses = []

    torch.cuda.empty_cache()
    scheduler.step()
    optimizer.zero_grad()
    
    for tr_batch_i, (X, target, sample_ids) in enumerate(train_dataloader):

        model.train() 

        X = X.to(device).float()  # [N, 3, H, W]
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
