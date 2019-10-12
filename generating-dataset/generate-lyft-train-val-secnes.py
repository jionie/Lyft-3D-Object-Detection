## This file is to generate train_scenes, val_scenes and test_scenes for lyft dataset

from lyft_dataset_sdk.lyftdataset import LyftDataset
import random

def generating_scenes():
    ############################ Set random seed for list shuffle
    SEED = 42
    random.seed(SEED)
    
    
    
    ############################ load lyft dataset 
    level5data = LyftDataset(data_path='.', \
        json_path='/media/jionie/my_disk/Kaggle/Lyft/input/3d-object-detection-for-autonomous-vehicles/train_data', verbose=True)
    level5data_test = LyftDataset(data_path='.', \
        json_path='/media/jionie/my_disk/Kaggle/Lyft/input/3d-object-detection-for-autonomous-vehicles/test_data', verbose=True)

    train_scenes = []
    val_scenes = []
    all_scenes = []
    test_scenes = []

    for sample in level5data.sample:
        all_scenes.append(sample["scene_token"])
        
    for sample in level5data_test.sample:
        test_scenes.append(sample["scene_token"])
        
    all_scenes = list(set(all_scenes))
    test_scenes = list(set(test_scenes))

    random.shuffle(all_scenes)
    train_scenes = all_scenes[:int(5 * len(all_scenes) / 6)]
    val_scenes = all_scenes[int(5 * len(all_scenes) / 6):]



    ############################ save splitting result
    with open("train_scenes.txt", "w") as f:
        for item in train_scenes:
            f.write("%s\n" % item)
            
    with open("val_scenes.txt", "w") as f:
        for item in val_scenes:
            f.write("%s\n" % item)
            
    with open("test_scenes.txt", "w") as f:
        for item in test_scenes:
            f.write("%s\n" % item)
            
            

if __name__ == "__main__":
    generating_scenes()