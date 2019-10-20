## This file is to generate train_scenes, val_scenes and test_scenes for lyft dataset

from lyft_dataset_sdk.lyftdataset import LyftDataset
from pathlib import Path
import random
import os
import glob

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--train_root_kitti', type=str, \
    default='/media/jionie/my_disk/Kaggle/Lyft/input/3d-object-detection-for-autonomous-vehicles/train_root/KITTI')
parser.add_argument('--test_root_kitti', type=str, \
    default='/media/jionie/my_disk/Kaggle/Lyft/input/3d-object-detection-for-autonomous-vehicles/test_root/KITTI')
args = parser.parse_args()

def generating_scenes_kitti(train_root_path, test_root_path):
    ############################ Set random seed for list shuffle
    SEED = 42
    random.seed(SEED)
    
    
    
    ############################ load calib
    train_scenes = []
    val_scenes = []
    all_scenes = glob.glob(train_root_path + "/calib/*.txt")
    test_scenes = glob.glob(test_root_path + "/calib/*.txt")
        
    all_scenes = list(set(all_scenes))
    test_scenes = list(set(test_scenes))

    random.shuffle(all_scenes)
    train_scenes = all_scenes[:int(5 * len(all_scenes) / 6)]
    val_scenes = all_scenes[int(5 * len(all_scenes) / 6):]

    if not os.path.exists(Path(train_root_path) / "ImageSets"):
        os.path.makdir(Path(train_root_path) / "ImageSets")
        
    if not os.path.exists(Path(test_root_path) / "ImageSets"):
        os.path.makdir(Path(test_root_path) / "ImageSets")

    ############################ save splitting result
    with open(Path(train_root_path) / "ImageSets/train.txt", "w") as f:
        for item in train_scenes:
            item = item.split("/")[-1].split(".")[0]
            f.write("%s\n" % item)
            
    with open(Path(train_root_path) / "ImageSets/val.txt", "w") as f:
        for item in val_scenes:
            item = item.split("/")[-1].split(".")[0]
            f.write("%s\n" % item)
            
    with open(Path(test_root_path) / "ImageSets/test.txt", "w") as f:
        for item in test_scenes:
            item = item.split("/")[-1].split(".")[0]
            f.write("%s\n" % item)
            
            

if __name__ == "__main__":
    generating_scenes_kitti(args.train_root_kitti, \
        args.test_root_kitti)