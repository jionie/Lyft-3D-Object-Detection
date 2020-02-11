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
Our final objective is to add eva support for storing and querying data for 3-D object detection and finish pipeline (data format changing, training and inference) for both Bird-Eye-View-Based method and PointRCNN based method (a 3D Liadar data based method). Right now, we finished pipeline for Bird-Eye-View-Based method. 

## Files for first code review
For first code review please focus on Bird-Eye-View-Based method, files related are in "unet_baseline" and "generating-dataset/generating-dataet-for-bev" folder. For "unet_baseline" folder, files in "models" and "utils" are some common used files for unet, if it's hard to understand you could view it as black box and code reivew for our pipeline instead.

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

## Files to look out for PointRCNN Method, still under developing
To use PointRCNN, we need to get KITTI format data by Lyft-sdk, please in "nuscenes-devkit" folder and run (please substitute "host-a011_lidar1_1233090652702363606.bin" in your "train_root/lidar" folder before transformation):
```bash
python3 -m export_kitti nuscenes_gt_to_kitti --lyft_dataroot "*/train_root" --table_folder "*/train_root/data" --store_dataroot "*/train_root/KITTI"
```
```bash
python3 -m export_kitti nuscenes_gt_to_kitti --lyft_dataroot "*/test_root" --table_folder "*/test_root/data" --store_dataroot "*/test_root/KITTI"
```
then, we need to get splits.txt like KITTI dataset, please in "generating-dataset" folder run:
```bash
python3 generate-lyft-train-val-secnes-kitti.py --train_root_kitti "*/train_root/KITTI" --test_root_kitti "*/test_root/KITTI"
```
Then we need to generate gt_database for PointRCNN, please in "PointRCNN" folder, follow the instruction to compile PointRCNN, then run:
```bash
python3 generate_gt_database.py --root_dir "*/train_root" --save_dir "*/train_root/KITTI/gt_database" --class_name Lyft --split train

```
```bash
python3 generate_gt_database.py --root_dir "*/train_root" --save_dir "*/train_root/KITTI/gt_database" --class_name Lyft --split train_val

```
```bash
python3 generate_gt_database.py --root_dir "*/train_root" --save_dir "*/train_root/KITTI/gt_database" --class_name Lyft --split val

```
```bash
python3 generate_gt_database.py --root_dir "*/test_root" --save_dir "*/test_root/KITTI/gt_database" --class_name Lyft --split test

```
Training rpn, (apex doesn't support, will have bugs), you can choose different optimizer in "cfgs/default.yaml", default is ranger + CosineAnealing, for PointRCNN repo default is adamonectcle. To train part of train set, please change train in "cfgs/default.yaml", otherwise whole dataset will cost large memory and you can't use large num_workers for dataloader (GPU utils will be low.
```bash
python3 train_rcnn.py --cfg_file cfgs/default.yaml --batch_size 16 --valid_batch_size 32 --train_mode rpn --data_root "*/train_root" --gt_database "*/train_root/KITTI/gt_database/train_part_1_gt_database_3level_emergency_vehicle.pkl" --output "*/train_root/KITTI/output" --pretrain_model "*/train_root/KITTI/output/rpn/default/ckpt/*.pth" --rpn_ckpt "*/train_root/KITTI/output/rpn/default/ckpt/*.pth"--epochs 100 --sub_epochs 5 --start_round * --start_part * --workers 4

```
We modified codes to iteratively training from 4 parts, you can change codes for your own parts, sub_epochs means each parts trains sub_epochs epochs then change to next part; one round means trained 4 parts. 

## License
[MIT](https://choosealicense.com/licenses/mit/)

