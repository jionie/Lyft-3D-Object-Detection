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
please in second folder run:
```bash
python3 generate-lyft-train-val-secnes.py --train_root_path=your_train_root_path --test_root_path=your_test_root_path
```
to generate scene splitting and run:
```bash
python3 create_data.py lyft_data_prep --root_path=your_train_root_path --version="lyft-trainval" --dataset_name="MyLyftDataset" --max_sweeps=10
```
```bash
python3 create_data.py lyft_data_prep --root_path=your_test_root_path --version="lyft-test" --dataset_name="MyLyftDataset" --max_sweeps=10
```
to generate dataset for second.

