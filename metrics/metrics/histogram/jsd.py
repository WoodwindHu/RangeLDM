import json
import torch
import numpy as np
#import open3d as o3d
import matplotlib.pyplot as plt

from metrics.histogram.mmd import load_point_cloud_xyz, load_point_cloud_xyz_nus
from .histogram import *
from .dist_helper import compute_mmd, gaussian, compute_mmd_sigma, gaussian_dist
import glob
import random 
from .hist_utils import *

def jsd_2d(p, q):
    from scipy.spatial.distance import jensenshannon
    return jensenshannon(p.flatten(), q.flatten())

def calculate_jsd_nus(sample_folder, from_bin=True):
    full_list_sample = glob.glob(f'{sample_folder}/*.bin')[:1000]
    model_histograms = []
    nus_histograms = []
    count = len(full_list_sample)
    nuScenes_path = os.environ.get('NUSCENES_DATASET')
    with open(os.path.join(nuScenes_path, 'v1.0-test/sample_data.json')) as f:
        sample_data = json.load(f)
    full_list = [os.path.join(nuScenes_path, x['filename']) 
                        for x in sample_data 
                        if 'sweeps/LIDAR_TOP' in x['filename']]
    seed = 0
    random.Random(seed).shuffle(full_list)
    full_list = full_list[0:count]
    if from_bin:
        for file in tqdm(full_list_sample):
            point_cloud = np.fromfile(file, dtype=np.float32).reshape(-1, 4)[:,:3]
            depth = np.linalg.norm(point_cloud[:,:3], 2, axis=1)
            mask = np.logical_and(depth>2.0, depth < 90.0)
            point_cloud = point_cloud[mask,:]
            hist = point_cloud_to_histogram(160, 100, point_cloud)[0]
            model_histograms.append(hist)
        
        # from .nuscenes_range_image import point_cloud_to_range_image_nuScenes
        # to_range = point_cloud_to_range_image_nuScenes()
        # to_range.mean = 50.
        # to_range.std = 50.                   
        for file in tqdm(full_list):
            point_cloud = load_point_cloud_xyz_nus(file, to_range=None)
            hist = point_cloud_to_histogram(160, 100, point_cloud)[0]
            nus_histograms.append(hist)
    else:
        raise NotImplementedError

    model_p = np.stack(model_histograms, axis=0)
    model_p = np.sum(model_p, axis=0)
    model_p = model_p / np.sum(model_p)

    nus_p = np.stack(nus_histograms, axis=0)
    nus_p = np.sum(nus_p, axis=0)
    nus_p = nus_p / np.sum(nus_p)

    jsd_score = jsd_2d(nus_p, model_p)

    return jsd_score

def calculate_jsd(sample_folder, from_bin=False):
    if from_bin:
        # model_samples = load_pts_to_range_images(sample_folder)
        full_list_sample = glob.glob(f'{sample_folder}/*.bin')
        model_histograms = []
        kitti_histograms = []

        for file in tqdm(full_list_sample):
            point_cloud = load_point_cloud_xyz(file)
            hist = point_cloud_to_histogram(160, 100, point_cloud)[0]
            model_histograms.append(hist)

        count = len(full_list_sample)
        full_list = glob.glob(os.environ.get('KITTI360_DATASET') +
                          '/data_3d_raw/2013_05_28_drive_0000_sync/velodyne_points/data/*')
        full_list.extend(glob.glob(os.environ.get('KITTI360_DATASET') +
                        '/data_3d_raw/2013_05_28_drive_0002_sync/velodyne_points/data/*'))
        seed = 0
        random.Random(seed).shuffle(full_list)
        full_list = full_list[0:count]
        # full_list = [os.environ.get('KITTI360_DATASET') + x for x in full_list]
        for file in tqdm(full_list):
            point_cloud = load_point_cloud_xyz(file)
            hist = point_cloud_to_histogram(160, 100, point_cloud)[0]
            kitti_histograms.append(hist)
    else:
        raise NotImplementedError

    model_p = np.stack(model_histograms, axis=0)
    model_p = np.sum(model_p, axis=0)
    model_p = model_p / np.sum(model_p)

    kitti_p = np.stack(kitti_histograms, axis=0)
    kitti_p = np.sum(kitti_p, axis=0)
    kitti_p = kitti_p / np.sum(kitti_p)

    jsd_score = jsd_2d(kitti_p, model_p)

    return jsd_score