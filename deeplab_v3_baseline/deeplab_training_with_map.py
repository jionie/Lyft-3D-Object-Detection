from datetime import datetime
from functools import partial
import glob
from multiprocessing import Pool
import argparse
from collections import OrderedDict
import warnings

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
import torch.utils.data as data
from torch.optim import lr_scheduler
from torch.optim.optimizer import Optimizer
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.utils.data.sampler import SubsetRandomSampler
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau, CosineAnnealingLR, _LRScheduler
import torch.nn.utils.weight_norm as weightNorm
import torch.nn.init as init
from torch.nn.parallel.data_parallel import data_parallel
import torchvision.models as models

from dataset.dataset import *
from tuils.tools import *
from tuils.lrs_scheduler import WarmRestart, warm_restart, AdamW
from tuils.loss_function import *
from tuils.ranger import *
import tuils.learning_schedules_fastai as lsf
from tuils.fastai_optim import OptimWrapper


from torch.utils.tensorboard import SummaryWriter
from apex import amp


parser = argparse.ArgumentParser(description="arg parser")
parser.add_argument('--model', type=str, default='deep_se101', required=False, help='specify the backbone model')
parser.add_argument('--optimizer', type=str, default='Ranger', required=False, help='specify the optimizer')
parser.add_argument("--lr_scheduler", type=str, default='WarmRestart', required=False, help="specify the lr scheduler")
parser.add_argument("--batch_size", type=int, default=8, required=False, help="specify the batch size for training")
parser.add_argument("--lr", type=float, default=1e-2, required=False, help="specify the initial lr for training")
parser.add_argument("--valid_batch_size", type=int, default=24, required=False, help="specify the batch size for validating")
parser.add_argument("--num_epoch", type=int, default=50, required=False, help="specify the total epoch")
parser.add_argument("--accumulation_steps", type=int, default=4, required=False, help="specify the accumulation steps")
parser.add_argument("--start_epoch", type=int, default=0, required=False, help="specify the start epoch for continue training")
parser.add_argument("--train_data_folder", type=str, default="/media/jionie/my_disk/Kaggle/Lyft/input/3d-object-detection-for-autonomous-vehicles/train_root/bev_data_with_map/", \
    required=False, help="specify the folder for training data")
parser.add_argument("--checkpoint_folder", type=str, default="/media/jionie/my_disk/Kaggle/Lyft/model/deeplab", \
    required=False, help="specify the folder for checkpoint")
parser.add_argument('--load_pretrain', action='store_true', default=False, help='whether to load pretrain model')

############################################################################## seed all
SEED = 10086
def seed_everything(seed=SEED):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
seed_everything(SEED)


############################################################################## calculate dice
def dice_overall(preds, targs):
    n = preds.shape[0]
    preds = preds.view(n, -1)
    targs = targs.view(n, -1)
    intersect = (preds * targs).sum(-1).float()
    union = (preds+targs).sum(-1).float()
    u0 = union==0
    intersect[u0] = 1
    union[u0] = 2
    return (2. * intersect / union)


def calc_dice(gt_seg, pred_seg):

    nom = 2 * np.sum(gt_seg * pred_seg)
    denom = np.sum(gt_seg) + np.sum(pred_seg)

    dice = float(nom) / float(denom)
    return dice


def calc_dice_all(preds_m, ys):
    dice_all = 0
    for i in range(preds_m.shape[0]):
        pred = preds_m[i,0,:,:]
        gt = ys[i,0,:,:]
#         print(np.sum(pred))
        if np.sum(gt) == 0 and np.sum(pred) == 0:
            dice_all = dice_all + 1
        elif np.sum(gt) == 0 and np.sum(pred) != 0:
            dice_all = dice_all
        else:
            dice_all = dice_all + calc_dice(gt, pred)
    return dice_all/preds_m.shape[0]


def children(m: nn.Module):
    return list(m.children())

def num_children(m: nn.Module):
    return len(children(m))

def load(model, pretrain_file, skip=[]):
    pretrain_state_dict = torch.load(pretrain_file)
    state_dict = model.state_dict()
    keys = list(state_dict.keys())
    for key in keys:
        if any(s in key for s in skip): continue
        try:
            state_dict[key] = pretrain_state_dict[key]
        except:
            print(key)
    model.load_state_dict(state_dict)
    
    return model

############################################################################## define model training
def deeplab_training(model_name,
                  optimizer_name,
                  lr_scheduler_name,
                  lr,
                  batch_size,
                  valid_batch_size,
                  num_epoch,
                  start_epoch,
                  accumulation_steps,
                  train_data_folder, 
                  checkpoint_folder,
                  load_pretrain
                  ):

    train_dataset, valid_dataset, train_dataloader, valid_dataloader = generate_dataset_loader(train_data_folder, batch_size, valid_batch_size, SEED)



    ############################################################################## define unet model with backbone
    def get_model(model_name="deep_se101", in_channel=6, num_classes=1, criterion=SoftDiceLoss_binary()):
        if model_name == 'deep_se50':
            from semantic_segmentation.network.deepv3 import DeepSRNX50V3PlusD_m1  # r
            model = DeepSRNX50V3PlusD_m1(in_channel=6, num_classes=num_classes, criterion=SoftDiceLoss_binary())
        elif model_name == 'deep_se101':
            from semantic_segmentation.network.deepv3 import DeepSRNX101V3PlusD_m1  # r
            model = DeepSRNX101V3PlusD_m1(in_channel=6, num_classes=num_classes, criterion=SoftDiceLoss_binary())
        elif model_name == 'WideResnet38':
            from semantic_segmentation.network.deepv3 import DeepWR38V3PlusD_m1  # r
            model = DeepWR38V3PlusD_m1(in_channel=6, num_classes=num_classes, criterion=SoftDiceLoss_binary())
        elif model_name == 'unet_ef3':
            from ef_unet import EfficientNet_3_unet
            model = EfficientNet_3_unet()
        elif model_name == 'unet_ef5':
            from ef_unet import EfficientNet_5_unet
            model = EfficientNet_5_unet()
        else:
            print('No model name in it')
            model = None
        return model

    # We weigh the loss for the 0 class lower to account for (some of) the big class imbalance.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    classes = ["car", "motorcycle", "bus", "bicycle", "truck", "pedestrian", "other_vehicle", "animal", "emergency_vehicle"]
    class_weights = torch.from_numpy(np.array([0.2] + [1.0]*len(classes), dtype=np.float32))
    class_weights = class_weights.to(device)



    ############################################################################### training parameters
    train_batch_size = batch_size
    checkpoint_filename = model_name + "_deeplab_checkpoint.pth"
    checkpoint_filepath = os.path.join(checkpoint_folder, checkpoint_filename)


    ############################################################################### model and optimizer
    model = get_model(model_name=model_name, in_channel=6, num_classes=len(classes)+1, criterion=SoftDiceLoss_binary())
    if (load_pretrain):
        model = load(model, checkpoint_filepath)
    model = model.to(device)

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
    elif lr_scheduler_name == "WarmRestart":
        scheduler = WarmRestart(optimizer, T_max=5, T_mult=1, eta_min=1e-6)
        lr_scheduler_each_iter = False
    else:
        raise NotImplementedError
        
    model, optimizer = amp.initialize(model, optimizer, opt_level="O1")
    
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
            
            prediction = model(X)  # [N, C, H, W]
            loss = F.cross_entropy(prediction, target, weight=class_weights)

            target = target.clone().unsqueeze_(1)
            one_hot = torch.cuda.FloatTensor(target.size(0), len(classes)+1, target.size(2), target.size(3)).zero_()
            target_one_hot = one_hot.scatter_(1, target.data, 1)
            loss += model.criterion(prediction, target_one_hot)

            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()

            #loss.backward()
        
            if ((tr_batch_i+1)%accumulation_steps==0):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0, norm_type=2)
                optimizer.step()
                optimizer.zero_grad()

                writer.add_scalar('train_loss', loss.item()*accumulation_steps, epoch*len(train_dataloader)*train_batch_size+tr_batch_i*train_batch_size)
            
            train_losses.append(loss.detach().cpu().numpy())

            if (tr_batch_i+1)%eval_step == 0:  

                with torch.no_grad():
                    
                    torch.cuda.empty_cache()

                    for val_batch_i, (X, target, sample_ids) in enumerate(valid_dataloader):

                        model.eval()

                        X = X.to(device).float()  # [N, 3, H, W]
                        target = target.to(device)  # [N, H, W] with class indices (0, 1)
                        prediction = model(X)  # [N, C, H, W]

                        ce_loss = F.cross_entropy(prediction, target, weight=class_weights).detach().cpu().numpy()

                        target = target.unsqueeze_(1)
                        one_hot = torch.cuda.FloatTensor(target.size(0), len(classes)+1, target.size(2), target.size(3)).zero_()
                        target_one_hot = one_hot.scatter_(1, target.data, 1)
                        loss = model.criterion(prediction, target_one_hot)

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
    
    deeplab_training(args.model, args.optimizer, args.lr_scheduler, args.lr, args.batch_size, args.valid_batch_size, \
                    args.num_epoch, args.start_epoch, args.accumulation_steps, args.train_data_folder, \
                    args.checkpoint_folder, args.load_pretrain)