from torchvision import transforms
import torchvision.datasets as datasets
from PIL import Image
import glob
import os
import torch.utils.data as data
import torch
import numpy as np
import cv2
from tqdm import tqdm
import random
import albumentations
# from tuils.preprocess import *
from torch.utils.data.dataloader import default_collate
from os.path import isfile
from skimage import feature
from scipy import ndimage
import json

import albumentations
from albumentations import torch as AT


SIZE = 336


def transform_train(image, mask):

    if random.random() < 0.5:
        image = albumentations.HorizontalFlip(p=1)(image=image)['image']
        mask = albumentations.HorizontalFlip(p=1)(image=mask)['image']

    if random.random() < 0.5:
        image = albumentations.Cutout(num_holes=1, max_h_size=32, max_w_size=32, p=1)(image=image)['image']
        mask = albumentations.Cutout(num_holes=1, max_h_size=32, max_w_size=32, p=1)(image=mask)['image']

    return image, mask

def transform_valid(image, mask):

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


def generate_dataset_loader(train_data_folder, train_batch_size, valid_batch_size):

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
    
    train_dataloader = torch.utils.data.DataLoader(train_dataset, train_batch_size, shuffle=True, num_workers=os.cpu_count()*2)
    valid_dataloader = torch.utils.data.DataLoader(valid_dataset, valid_batch_size, shuffle=False, num_workers=os.cpu_count()*2)
    
    return train_dataset, valid_dataset, train_dataloader, valid_dataloader
