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
please in second folder and run

python3 create_data.py lyft_data_prep --root_path="/media/jionie/my_disk/Kaggle/Lyft/input/3d-object-detection-for-autonomous-vehicles/train_root" --version="lyft-trainval" --dataset_name="MyLyftDataset" --max_sweeps=10

python3 create_data.py lyft_data_prep --root_path="/media/jionie/my_disk/Kaggle/Lyft/input/3d-object-detection-for-autonomous-vehicles/test_root" --version="lyft-test" --dataset_name="MyLyftDataset" --max_sweeps=10

to generate dataset for second.

