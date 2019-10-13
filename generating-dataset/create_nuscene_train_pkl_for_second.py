import json
import pickle
import time
import random
from copy import deepcopy
from functools import partial
from pathlib import Path
import subprocess

import fire
import numpy as np
from pyquaternion import Quaternion

from lyft_dataset_sdk.lyftdataset import LyftDataset

def _get_available_scenes(nusc):
    available_scenes = []
    print("total scene num:", len(nusc.scene))
    for scene in nusc.scene:
        scene_token = scene["token"]
        scene_rec = nusc.get('scene', scene_token)
        sample_rec = nusc.get('sample', scene_rec['first_sample_token'])
        sd_rec = nusc.get('sample_data', sample_rec['data']["LIDAR_TOP"])
        has_more_frames = True
        scene_not_exist = False
        while has_more_frames:
            lidar_path, boxes, _ = nusc.get_sample_data(sd_rec['token'])
            if not Path(lidar_path).exists():
                scene_not_exist = True
                break
            else:
                break
            if not sd_rec['next'] == "":
                sd_rec = nusc.get('sample_data', sd_rec['next'])
            else:
                has_more_frames = False
        if scene_not_exist:
            continue
        available_scenes.append(scene)
    print("exist scene num:", len(available_scenes))
    return available_scenes

def _fill_trainval_infos(nusc,
                         train_scenes,
                         val_scenes,
                         test=False,
                         max_sweeps=10):
    
    NameMapping = {
        'movable_object.barrier': 'barrier',
        'vehicle.bicycle': 'bicycle',
        'vehicle.bus.bendy': 'bus',
        'vehicle.bus.rigid': 'bus',
        'vehicle.car': 'car',
        'vehicle.construction': 'construction_vehicle',
        'vehicle.motorcycle': 'motorcycle',
        'human.pedestrian.adult': 'pedestrian',
        'human.pedestrian.child': 'pedestrian',
        'human.pedestrian.construction_worker': 'pedestrian',
        'human.pedestrian.police_officer': 'pedestrian',
        'movable_object.trafficcone': 'traffic_cone',
        'vehicle.trailer': 'trailer',
        'vehicle.truck': 'truck'
    }
    
    train_nusc_infos = []
    val_nusc_infos = []
    
    for sample in nusc.sample:
        lidar_token = sample["data"]["LIDAR_TOP"]
        cam_front_token = sample["data"]["CAM_FRONT"]
        sd_rec = nusc.get('sample_data', sample['data']["LIDAR_TOP"])
        cs_record = nusc.get('calibrated_sensor',
                             sd_rec['calibrated_sensor_token'])
        pose_record = nusc.get('ego_pose', sd_rec['ego_pose_token'])
        lidar_path, boxes, _ = nusc.get_sample_data(lidar_token)

        cam_path, _, cam_intrinsic = nusc.get_sample_data(cam_front_token)
        assert Path(lidar_path).exists(), (
            "you must download all trainval data, key-frame only dataset performs far worse than sweeps."
        )
        info = {
            "lidar_path": lidar_path,
            "cam_front_path": cam_path,
            "token": sample["token"],
            "sweeps": [],
            "lidar2ego_translation": cs_record['translation'],
            "lidar2ego_rotation": cs_record['rotation'],
            "ego2global_translation": pose_record['translation'],
            "ego2global_rotation": pose_record['rotation'],
            "timestamp": sample["timestamp"],
        }

        l2e_r = info["lidar2ego_rotation"]
        l2e_t = info["lidar2ego_translation"]
        e2g_r = info["ego2global_rotation"]
        e2g_t = info["ego2global_translation"]
        l2e_r_mat = Quaternion(l2e_r).rotation_matrix
        e2g_r_mat = Quaternion(e2g_r).rotation_matrix

        sd_rec = nusc.get('sample_data', sample['data']["LIDAR_TOP"])
        sweeps = []
        while len(sweeps) < max_sweeps:
            if not sd_rec['prev'] == "":
                sd_rec = nusc.get('sample_data', sd_rec['prev'])
                cs_record = nusc.get('calibrated_sensor',
                                     sd_rec['calibrated_sensor_token'])
                pose_record = nusc.get('ego_pose', sd_rec['ego_pose_token'])
                lidar_path = nusc.get_sample_data_path(sd_rec['token'])
                sweep = {
                    "lidar_path": lidar_path,
                    "sample_data_token": sd_rec['token'],
                    "lidar2ego_translation": cs_record['translation'],
                    "lidar2ego_rotation": cs_record['rotation'],
                    "ego2global_translation": pose_record['translation'],
                    "ego2global_rotation": pose_record['rotation'],
                    "timestamp": sd_rec["timestamp"]
                }
                l2e_r_s = sweep["lidar2ego_rotation"]
                l2e_t_s = sweep["lidar2ego_translation"]
                e2g_r_s = sweep["ego2global_rotation"]
                e2g_t_s = sweep["ego2global_translation"]
                # sweep->ego->global->ego'->lidar
                l2e_r_s_mat = Quaternion(l2e_r_s).rotation_matrix
                e2g_r_s_mat = Quaternion(e2g_r_s).rotation_matrix

                R = (l2e_r_s_mat.T @ e2g_r_s_mat.T) @ (
                    np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T)
                T = (l2e_t_s @ e2g_r_s_mat.T + e2g_t_s) @ (
                    np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T)
                T -= e2g_t @ (np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(
                    l2e_r_mat).T) + l2e_t @ np.linalg.inv(l2e_r_mat).T
                sweep["sweep2lidar_rotation"] = R.T  # points @ R.T + T
                sweep["sweep2lidar_translation"] = T
                sweeps.append(sweep)
            else:
                break
        info["sweeps"] = sweeps
        if not test:
            annotations = [
                nusc.get('sample_annotation', token)
                for token in sample['anns']
            ]
            locs = np.array([b.center for b in boxes]).reshape(-1, 3)
            dims = np.array([b.wlh for b in boxes]).reshape(-1, 3)
            rots = np.array([b.orientation.yaw_pitch_roll[0]
                             for b in boxes]).reshape(-1, 1)
            velocity = np.array(
                [nusc.box_velocity(token)[:2] for token in sample['anns']])
            # convert velo from global to lidar
            for i in range(len(boxes)):
                velo = np.array([*velocity[i], 0.0])
                velo = velo @ np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(
                    l2e_r_mat).T
                velocity[i] = velo[:2]

            names = [b.name for b in boxes]
            for i in range(len(names)):
                if names[i] in NameMapping:
                    names[i] = NameMapping[names[i]]
            names = np.array(names)
            # we need to convert rot to SECOND format.
            # change the rot format will break all checkpoint, so...
            gt_boxes = np.concatenate([locs, dims, -rots - np.pi / 2], axis=1)
            assert len(gt_boxes) == len(
                annotations), f"{len(gt_boxes)}, {len(annotations)}"
            info["gt_boxes"] = gt_boxes
            info["gt_names"] = names
            info["gt_velocity"] = velocity.reshape(-1, 2)
            info["num_lidar_pts"] = np.array(
                [a["num_lidar_pts"] for a in annotations])
            info["num_radar_pts"] = np.array(
                [a["num_radar_pts"] for a in annotations])
        if sample["scene_token"] in train_scenes:
            train_nusc_infos.append(info)
        else:
            val_nusc_infos.append(info)
    return train_nusc_infos, val_nusc_infos

def create_lyft_infos(root_path, version="lyft-trainval", max_sweeps=10):
    
    from nuscenes.nuscenes import NuScenes
    from nuscenes.utils import splits
    
    if (version in ["v1.0-trainval", "v1.0-test", "v1.0-mini"]):
        nusc = NuScenes(version=version, dataroot=root_path, verbose=True)
    else:
        nusc = LyftDataset(data_path=root_path, json_path=Path(root_path) / "data", verbose=True)
  
    available_vers = ["v1.0-trainval", "v1.0-test", "v1.0-mini", "lyft-trainval", "lyft-test"]
    assert version in available_vers
    if version == "v1.0-trainval":
        train_scenes = splits.train
        val_scenes = splits.val
    elif version == "v1.0-test":
        train_scenes = splits.test
        val_scenes = []
    elif version == "v1.0-mini":
        train_scenes = splits.mini_train
        val_scenes = splits.mini_val
    elif version == "lyft-trainval":
        with open(Path(root_path) / "train_scenes.txt") as f:
            train_scenes = f.read().splitlines()
            
        with open(Path(root_path) / "val_scenes.txt") as f:
            val_scenes = f.read().splitlines()

    elif version == "lyft-test":
        with open(Path(root_path) / "test_scenes.txt") as f:
            train_scenes = f.read().splitlines()
        val_scenes = []
    else:
        raise ValueError("unknown")
    test = (version == "lyft-test") 
    root_path = Path(root_path)
    
    if (version in ["v1.0-trainval", "v1.0-test", "v1.0-mini"]):
        # filter exist scenes. you may only download part of dataset.
        available_scenes = _get_available_scenes(nusc)
        available_scene_names = [s["name"] for s in available_scenes]
        train_scenes = list(
            filter(lambda x: x in available_scene_names, train_scenes))
        val_scenes = list(filter(lambda x: x in available_scene_names, val_scenes))
        train_scenes = set([
            available_scenes[available_scene_names.index(s)]["token"]
            for s in train_scenes
        ])
        val_scenes = set([
            available_scenes[available_scene_names.index(s)]["token"]
            for s in val_scenes
        ])
        
    if test:
        print(f"test scene: {len(train_scenes)}")
    else:
        print(
            f"train scene: {len(train_scenes)}, val scene: {len(val_scenes)}")
    train_nusc_infos, val_nusc_infos = _fill_trainval_infos(
        nusc, train_scenes, val_scenes, test, max_sweeps=max_sweeps)
    metadata = {
        "version": version,
    }
    if test:
        print(f"test sample: {len(train_nusc_infos)}")
        data = {
            "infos": train_nusc_infos,
            "metadata": metadata,
        }
        with open(Path(root_path) /  "infos_test.pkl", 'wb') as f:
            pickle.dump(data, f)
    else:
        print(
            f"train sample: {len(train_nusc_infos)}, val sample: {len(val_nusc_infos)}"
        )
        data = {
            "infos": train_nusc_infos,
            "metadata": metadata,
        }
        with open(Path(root_path) /  "infos_train.pkl", 'wb') as f:
            pickle.dump(data, f)
        data["infos"] = val_nusc_infos
        with open(Path(root_path) /  "infos_val.pkl", 'wb') as f:
            pickle.dump(data, f)
            
if __name__ == "__main__":
    create_lyft_infos(root_path='/media/jionie/my_disk/Kaggle/Lyft/input/3d-object-detection-for-autonomous-vehicles/train_root', \
        version="lyft-trainval", max_sweeps=10)