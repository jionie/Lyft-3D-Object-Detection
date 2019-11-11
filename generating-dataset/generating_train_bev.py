from lyft_dataset_sdk.lyftdataset import LyftDataset
from datetime import datetime
from functools import partial
import glob
from multiprocessing import Pool

# Disable multiprocesing for numpy/opencv. We already multiprocess ourselves, this would mean every subprocess produces
# even more threads which would lead to a lot of context switching, slowing things down a lot.
import os
os.environ["OMP_NUM_THREADS"] = "1"

import matplotlib.pyplot as plt

import pandas as pd
import cv2
from PIL import Image
import numpy as np
from tqdm import tqdm, tqdm_notebook
import scipy
import scipy.ndimage
import scipy.special
from scipy.spatial.transform import Rotation as R
from pathlib import Path

from lyft_dataset_sdk.lyftdataset import LyftDataset,LyftDatasetExplorer
from lyft_dataset_sdk.utils.data_classes import LidarPointCloud, Box, Quaternion
from lyft_dataset_sdk.utils.geometry_utils import view_points, transform_matrix
import time
from lyft_dataset_sdk.utils.map_mask import MapMask
from functools import partial
import glob
from multiprocessing import Pool


def generating_train_bev():
    # Some hyperparameters we'll need to define for the system
    voxel_size = (0.4, 0.4, 1.5)
    z_offset = -2.0
    bev_shape = (336, 336, 3)

    # We scale down each box so they are more separated when projected into our coarse voxel space.
    box_scale = 0.8

    NUM_WORKERS = os.cpu_count() * 3


    def create_transformation_matrix_to_voxel_space(shape, voxel_size, offset):
        """
        Constructs a transformation matrix given an output voxel shape such that (0,0,0) ends up in the center.
        Voxel_size defines how large every voxel is in world coordinate, (1,1,1) would be the same as Minecraft voxels.
        
        An offset per axis in world coordinates (metric) can be provided, this is useful for Z (up-down) in lidar points.
        """
        
        shape, voxel_size, offset = np.array(shape), np.array(voxel_size), np.array(offset)
        
        tm = np.eye(4, dtype=np.float32)
        translation = shape/2 + offset/voxel_size
        
        tm = tm * np.array(np.hstack((1/voxel_size, [1])))
        tm[:3, 3] = np.transpose(translation)
        return tm

    def transform_points(points, transf_matrix):
        """
        Transform (3,N) or (4,N) points using transformation matrix.
        """
        if points.shape[0] not in [3,4]:
            raise Exception("Points input should be (3,N) or (4,N) shape, received {}".format(points.shape))
        return transf_matrix.dot(np.vstack((points[:3, :], np.ones(points.shape[1]))))[:3, :]


    def car_to_voxel_coords(points, shape, voxel_size, z_offset=0):
        if len(shape) != 3:
            raise Exception("Voxel volume shape should be 3 dimensions (x,y,z)")
            
        if len(points.shape) != 2 or points.shape[0] not in [3, 4]:
            raise Exception("Input points should be (3,N) or (4,N) in shape, found {}".format(points.shape))

        tm = create_transformation_matrix_to_voxel_space(shape, voxel_size, (0, 0, z_offset))
        p = transform_points(points, tm)
        return p

    def create_voxel_pointcloud(points, shape, voxel_size=(0.5,0.5,1), z_offset=0):

        points_voxel_coords = car_to_voxel_coords(points.copy(), shape, voxel_size, z_offset)
        points_voxel_coords = points_voxel_coords[:3].transpose(1,0)
        points_voxel_coords = np.int0(points_voxel_coords)
        
        bev = np.zeros(shape, dtype=np.float32)
        bev_shape = np.array(shape)

        within_bounds = (np.all(points_voxel_coords >= 0, axis=1) * np.all(points_voxel_coords < bev_shape, axis=1))
        
        points_voxel_coords = points_voxel_coords[within_bounds]
        coord, count = np.unique(points_voxel_coords, axis=0, return_counts=True)
            
        # Note X and Y are flipped:
        bev[coord[:,1], coord[:,0], coord[:,2]] = count
        
        return bev

    def normalize_voxel_intensities(bev, max_intensity=16):
        return (bev/max_intensity).clip(0,1)


    def move_boxes_to_car_space(boxes, ego_pose):
        """
        Move boxes from world space to car space.
        Note: mutates input boxes.
        """
        translation = -np.array(ego_pose['translation'])
        rotation = Quaternion(ego_pose['rotation']).inverse
        
        for box in boxes:
            # Bring box to car space
            box.translate(translation)
            box.rotate(rotation)
            
    def scale_boxes(boxes, factor):
        """
        Note: mutates input boxes
        """
        for box in boxes:
            box.wlh = box.wlh * factor

    def draw_boxes(im, voxel_size, boxes, classes, z_offset=0.0):
        for box in boxes:
            # We only care about the bottom corners
            corners = box.bottom_corners()
            corners_voxel = car_to_voxel_coords(corners, im.shape, voxel_size, z_offset).transpose(1,0)
            corners_voxel = corners_voxel[:,:2] # Drop z coord

            class_color = classes.index(box.name) + 1
            
            if class_color == 0:
                raise Exception("Unknown class: {}".format(box.name))

            cv2.drawContours(im, np.int0([corners_voxel]), 0, (class_color, class_color, class_color), -1)
            
    def get_semantic_map_around_ego(map_mask, ego_pose, voxel_size, output_shape):
    
        def crop_image(image: np.array,
                            x_px: int,
                            y_px: int,
                            axes_limit_px: int) -> np.array:
                    x_min = int(x_px - axes_limit_px)
                    x_max = int(x_px + axes_limit_px)
                    y_min = int(y_px - axes_limit_px)
                    y_max = int(y_px + axes_limit_px)

                    cropped_image = image[y_min:y_max, x_min:x_max]

                    return cropped_image

        pixel_coords = map_mask.to_pixel_coords(ego_pose['translation'][0], ego_pose['translation'][1])

        extent = voxel_size*output_shape[0]*0.5
        scaled_limit_px = int(extent * (1.0 / (map_mask.resolution)))
        mask_raster = map_mask.mask()

        cropped = crop_image(mask_raster, pixel_coords[0], pixel_coords[1], int(scaled_limit_px * np.sqrt(2)))

        ypr_rad = Quaternion(ego_pose['rotation']).yaw_pitch_roll
        yaw_deg = -np.degrees(ypr_rad[0])

        rotated_cropped = np.array(Image.fromarray(cropped).rotate(yaw_deg))
        ego_centric_map = crop_image(rotated_cropped, rotated_cropped.shape[1] / 2, rotated_cropped.shape[0] / 2,
                                    scaled_limit_px)[::-1]
        
        ego_centric_map = cv2.resize(ego_centric_map, output_shape[:2], cv2.INTER_NEAREST)
        return ego_centric_map.astype(np.float32)/255


    def prepare_training_data_for_scene(sample_token, level5data, output_folder,
                                    bev_shape=bev_shape, voxel_size=voxel_size, z_offset=z_offset,
                                    box_scale=box_scale):
        """
        Given a sample token (in a scene), output rasterized input volumes in birds-eye-view perspective.

        """
        if (os.path.exists(os.path.join(output_folder, "{}_target.png".format(sample_token)))):
            print("file processed, skip")

        else:

            sample = level5data.get("sample", sample_token)
            
            sample_lidar_token = sample["data"]["LIDAR_TOP"]
            lidar_data = level5data.get("sample_data", sample_lidar_token)
            lidar_filepath = level5data.get_sample_data_path(sample_lidar_token)
            
            

            ego_pose = level5data.get("ego_pose", lidar_data["ego_pose_token"])
            calibrated_sensor = level5data.get("calibrated_sensor", lidar_data["calibrated_sensor_token"])
            


            global_from_car = transform_matrix(ego_pose['translation'],
                                            Quaternion(ego_pose['rotation']), inverse=False)
            

            car_from_sensor = transform_matrix(calibrated_sensor['translation'], Quaternion(calibrated_sensor['rotation']),
                                                inverse=False)

            try:
                lidar_pointcloud = LidarPointCloud.from_file(lidar_filepath)
                lidar_pointcloud.transform(car_from_sensor)
                bev = create_voxel_pointcloud(lidar_pointcloud.points, bev_shape, voxel_size=voxel_size, z_offset=z_offset)
                bev = normalize_voxel_intensities(bev)

                bev_im = np.round(bev*255).astype(np.uint8)

                cv2.imwrite(os.path.join(output_folder, "{}_input.png".format(sample_token)), bev_im)

                boxes = level5data.get_boxes(sample_lidar_token)
                target = np.zeros_like(bev)

                move_boxes_to_car_space(boxes, ego_pose)
                scale_boxes(boxes, box_scale)
                draw_boxes(target, voxel_size, boxes=boxes, classes=classes, z_offset=z_offset)
                target_im = target[:,:,0] # take one channel only

                cv2.imwrite(os.path.join(output_folder, "{}_target.png".format(sample_token)), target_im)
                
                map_mask = level5data.map[0]["mask"]
                semantic_im = get_semantic_map_around_ego(map_mask, ego_pose, voxel_size[0], target_im.shape)
                semantic_im = np.round(semantic_im*255).astype(np.uint8)
                cv2.imwrite(os.path.join(output_folder, "{}_map.png".format(sample_token)), semantic_im)

            except Exception as e:
                print ("Failed to load Lidar Pointcloud for {}: {}:".format(sample_token, e))

    train_data_folder = '/media/jionie/my_disk/Kaggle/Lyft/input/3d-object-detection-for-autonomous-vehicles/train_root/bev_data_with_map'

    level5data_train = LyftDataset(data_path='/media/jionie/my_disk/Kaggle/Lyft/input/3d-object-detection-for-autonomous-vehicles/train_root', \
        json_path='/media/jionie/my_disk/Kaggle/Lyft/input/3d-object-detection-for-autonomous-vehicles/train_root/data', verbose=True)
    classes = ["car", "motorcycle", "bus", "bicycle", "truck", "pedestrian", "other_vehicle", "animal", "emergency_vehicle"]


    train = pd.read_csv("/media/jionie/my_disk/Kaggle/Lyft/input/3d-object-detection-for-autonomous-vehicles/train_root/train.csv")

    for i in range(train.shape[0]):

        if (i % 100 == 0):
            print("processing: ", i, " of ", train.shape[0], " samples.")

        token = train.loc[i,'Id']
        prepare_training_data_for_scene(token, level5data_train, train_data_folder)
        

if __name__ == "__main__":
    generating_train_bev()