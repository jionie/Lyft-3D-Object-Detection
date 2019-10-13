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
```bash
python3 create_data.py lyft_data_prep --root_path=train_root_path --version="lyft-trainval" --dataset_name="MyLyftDataset" --max_sweeps=10
```
```bash
python3 create_data.py lyft_data_prep --root_path=test_root_path --version="lyft-test" --dataset_name="MyLyftDataset" --max_sweeps=10
```
to generate dataset for second.

