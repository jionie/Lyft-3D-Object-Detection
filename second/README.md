# SECOND for Lyft object detection
Source codes from [second.pytorch](https://github.com/pyaf/second.pytorch)

Modifications:

* Support for Lyft's level 5 dataset.
* Some small tweaks to get the nuscenes version working for lyft.
* The evaluation code is modified to include competition's evaluation metric which uses a range of IoU thresholds for mAP (unlike the original metric which used a range distance thresholds).
* second/notebooks/*.ipynb files contain my submission and inference testing code, it needs some cleanup

ONLY support python 3.6+, pytorch 1.0.0+. Tested in Ubuntu 16.04/18.04/Windows 10.

## Install

### 1. Clone code

```bash
git clone https://github.com/pyaf/second.pytorch.git
cd ./second.pytorch/second
```

### 2. Install dependencies

It is recommend to use Anaconda package manager.

```bash
conda install scikit-image scipy numba pillow matplotlib
```

```bash
pip install fire tensorboardX protobuf opencv-python
```

Follow instructions in [spconv](https://github.com/traveller59/spconv) to install spconv.

If you want to train with fp16 mixed precision (train faster in RTX series, Titan V/RTX and Tesla V100, but I only have 1080Ti), you need to install [apex](https://github.com/NVIDIA/apex).

### 3. add second.pytorch/ to PYTHONPATH

Add following line to your .bashrc, update the path accordingly

`export PYTHONPATH="${PYTHONPATH}:/media/ags/DATA/CODE/kaggle/lyft-3d-object-detection/second.pytorch"`

## Prepare dataset

* [Lyft]() Dataset preparation

Download Lyft dataset:

```plain
└── LYFT_TRAINVAL_DATASET_ROOT
       ├── lidar         <-- lidar files
       ├── maps          <-- unused
       ├── images        <-- unused
       ├── data          <-- metadata and annotations
       └── v1.0-trainval <-- softlink to `data`

└── LYFT_TEST_DATASET_ROOT
       ├── lidar         <-- lidar files
       ├── maps          <-- unused
       ├── images        <-- unused
       ├── data          <-- metadata and annotations
       └── v1.0-test     <-- softlink to `data`
```

NOTE: `v1.0-*` folders in train/test folders are soft links to corresponding `data` folders

```bash
python create_data.py nuscenes_data_prep --root_path=LYFT_TRAINVAL_DATASET_ROOT  --version="v1.0-trainval" --dataset_name="NuScenesDataset" --max_sweeps=10
python create_data.py nuscenes_data_prep --root_path=LYFT_TEST_DATASET_ROOT  --version="v1.0-test" --dataset_name="NuScenesDataset" --max_sweeps=10
```

`LYFT_TRAINVAL_DATASET_ROOT` are full path to train set of the dataset, similaryly for `LYFT_TEST_DATASET_ROOT`. Also you need to generate ground truth info by "second/notebooks/prepare.ipynb"


Rest of this readme is from original second implementation.

* Modify config file

There is some path need to be configured in config file:

```bash
train_input_reader: {
  ...
  database_sampler {
    database_info_path: "/path/to/dataset_dbinfos_train.pkl"
    ...
  }
  dataset: {
    dataset_class_name: "DATASET_NAME"
    kitti_info_path: "/path/to/dataset_infos_train.pkl"
    kitti_root_path: "DATASET_ROOT"
  }
}
...
eval_input_reader: {
  ...
  dataset: {
    dataset_class_name: "DATASET_NAME"
    kitti_info_path: "/path/to/dataset_infos_val.pkl"
    kitti_root_path: "DATASET_ROOT"
  }
}
```

## Usage

#### train with single GPU

```bash
python ./pytorch/train.py train --config_path=./configs/all.pp.lowa.config --model_dir=/path/to/model_dir
```

#### train with multiple GPU (need test, I only have one GPU)

Assume you have 4 GPUs and want to train with 3 GPUs:

```bash
CUDA_VISIBLE_DEVICES=0,1,3 python ./pytorch/train.py train --config_path=./configs/all.pp.lowa.config --model_dir=/path/to/model_dir --multi_gpu=True
```

Note: The batch_size and num_workers in config file is per-GPU, if you use multi-gpu, they will be multiplied by number of GPUs. Don't modify them manually.

You need to modify total step in config file. For example, 50 epochs = 15500 steps for car.lite.config and single GPU, if you use 4 GPUs, you need to divide ```steps``` and ```steps_per_eval``` by 4.

#### train with fp16 (mixed precision)

Modify config file, set enable_mixed_precision to true.

* Make sure "/path/to/model_dir" doesn't exist if you want to train new model. A new directory will be created if the model_dir doesn't exist, otherwise will read checkpoints in it.

* training process use batchsize=6 as default for 1080Ti, you need to reduce batchsize if your GPU has less memory.

* Currently only support single GPU training, but train a model only needs 20 hours (165 epoch) in a single 1080Ti and only needs 50 epoch to reach 78.3 AP with super converge in car moderate 3D in Kitti validation dateset.

### evaluate

```bash
python ./pytorch/train.py evaluate --config_path=./configs/all.pp.lowa.config --model_dir=/path/to/model_dir --measure_time=True --batch_size=1
```

* detection result will saved as a result.pkl file in model_dir/eval_results/step_xxx or save as official KITTI label format if you use --pickle_result=False.

### inference

You can simply use "submission.ipynb" to get video result and csv result to submit to kaggle.


## Try Kitti Viewer Web

I've modified original Kitti viewer to get it working for lyft inference, do give it a try after training.

### Major step

1. run ```python ./kittiviewer/backend/main.py main --port=xxxx``` in your server/local.

2. run ```cd ./kittiviewer/frontend && python -m http.server``` to launch a local web server.

3. open your browser and enter your frontend url (e.g. http://127.0.0.1:8000, default]).

4. input backend url (e.g. http://127.0.0.1:16666)

5. input root path, info path and det path (optional)

6. click load, loadDet (optional), input image index in center bottom of screen and press Enter.

### Inference step

Firstly the load button must be clicked and load successfully.

1. input checkpointPath and configPath.

2. click buildNet.

3. click inference.

![GuidePic](https://raw.githubusercontent.com/traveller59/second.pytorch/master/images/viewerweb.png)



## Try Kitti Viewer (Deprecated)

You should use kitti viewer based on pyqt and pyqtgraph to check data before training.

run ```python ./kittiviewer/viewer.py```, check following picture to use kitti viewer:
![GuidePic](https://raw.githubusercontent.com/traveller59/second.pytorch/master/images/simpleguide.png)

## Concepts


* Kitti lidar box

A kitti lidar box is consist of 7 elements: [x, y, z, w, l, h, rz], see figure.

![Kitti Box Image](https://raw.githubusercontent.com/traveller59/second.pytorch/master/images/kittibox.png)

All training and inference code use kitti box format. So we need to convert other format to KITTI format before training.

* Kitti camera box

A kitti camera box is consist of 7 elements: [x, y, z, l, h, w, ry].

