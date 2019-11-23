# Lyft-3D-Object-Detection
This repository contains codes for 3-D object detection for Autonomous Vehicles. Including Bird-Eye-View-Based method and PointRCNN method (third party library). This repository is a copy version of our private repository which we'll make it public after competition ends, thus this repository only contains few commits, our team contribute to this repository equally. 

## Structure for data
please arrange input folder as 
```plain
input
└── train_root
|      ├── data
|      ├── images
|      ├── lidar
|      ├── maps
|      ├── train.csv
|
└── test_root
       ├── data
       ├── images
       ├── lidar
       ├── maps
       ├── sample_submission.csv
model
└── unet
```
## Our plan
Our final objective is to add eva support for storing and querying data for 3-D object detection and finish pipeline (data format changing, training and inference) for both Bird-Eye-View-Based method and PointPilliars based method (a 3D Liadar data based method). Right now, we finished pipeline for Bird-Eye-View-Based method. 

## Files for first code review
For first code review please focus on Bird-Eye-View-Based method, files related are in "unet_baseline" and "generating-dataset/generating-dataet-for-bev" folder. The final result in kaggle is 0.045.

## Files to look out for Unet-Based Method
For Unet-Based method, we need to generate Bird Eye View format data first, please enter folder "generating-dataset/generating-dataset" and run:
```bash
python3 generating_train_bev.py

```
```bash
python3 generating_test_bev.py

```
Once we finished data transformation, we could enter folder "unet_baseline" and run:
```bash
python3 unet-training-with-map --model seresnext101 --optimizer Ranger --lr_scheduler adamonecycle --batch_size 32 --valid_batch_size 64 --num_epoch 50 --accumulation_steps 4 
--start_epoch 0 --train_data_folder "*/bev_data_with_map/" --checkpoint_folder "*/model/unet"
```
Where the "*/bev_data_with_map/" and "*/model/unet" are the path to generated dataset. After training, you could use "unet-inference-with-mask-visualization.ipynb" to inference and get visualization of one scene prediction.

## Files to look out for PointPilliars Method
To use PointPilliars with second, please follow "README.md" in "second" folder. We modified codes from existing second library for this task, without parameter tuning, the final result in kaggle is 0.049, you can get higher with changing config files.

## License
[MIT](https://choosealicense.com/licenses/mit/)


