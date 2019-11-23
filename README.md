# Lyft-3D-Object-Detection
please arrange input folder as 
```plain
input/3d-object-detection-for-autonomous-vehicles
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
```
We plan to use PointRCNN first, so we need to get KITTI format data by Lyft-sdk, please in "nuscenes-devkit" folder and run (please substitute "host-a011_lidar1_1233090652702363606.bin" in your "train_root/lidar" folder before transformation):
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

Failed with PointRCNN, we decide to move to second with PointPillars, first we used same KITTI format data, then we cd "second" folder run:
```bash
python3 train.py train --config_path="*/second/second/configs/lyft/all.fhd.config" --model_dir= "*/model/second/PointPillars"
```
