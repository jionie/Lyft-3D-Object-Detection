# Lyft-3D-Object-Detection
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
python3 generate_gt_database.py --root_dir "*/train_root" --save_dir "*/train_root/KITTI/gt_database" --class_name Lyft --split val

```
```bash
python3 generate_gt_database.py --root_dir "*/test_root" --save_dir "*/test_root/KITTI/gt_database" --class_name Lyft --split test

```

