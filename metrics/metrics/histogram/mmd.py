
import json
import torch
import numpy as np
#import open3d as o3d
import matplotlib.pyplot as plt
from .histogram import *
from .dist_helper import compute_mmd, gaussian, compute_mmd_sigma, gaussian_dist
import glob
import random 
from .hist_utils import *
from tqdm import tqdm
from sklearn import metrics
import torch

def mmd_rbf(X, Y, sigma=0.5):
    X = np.array([x/np.sum(x) for x in X]).astype(np.float32)
    Y = np.array([y/np.sum(y) for y in Y]).astype(np.float32)
    X = torch.from_numpy(X).cuda()
    Y = torch.from_numpy(Y).cuda()
    XX = torch.zeros((X.shape[0], X.shape[0])).cuda()
    YY = torch.zeros((X.shape[0], X.shape[0])).cuda()
    XY = torch.zeros((X.shape[0], X.shape[0])).cuda()
    for i in tqdm(range(X.shape[0])):
        XX[:,i:i+1] = torch.linalg.norm(X[:,None,:,:] - X[None, i:i+1,:,:],2, dim=(-2,-1))
    XX = torch.exp(-XX * XX / (2 * sigma * sigma)).mean()
    print('s1: ', XX.item())
    for i in tqdm(range(Y.shape[0])):
        YY[:,i:i+1] = torch.linalg.norm(Y[:,None,:,:] - Y[None, i:i+1,:,:],2, dim=(-2,-1))
    YY = torch.exp(-YY * YY / (2 * sigma * sigma)).mean()
    print('s2: ', YY.item())
    for i in tqdm(range(Y.shape[0])):
        XY[:,i:i+1] = torch.linalg.norm(X[:,None,:,:] - Y[None, i:i+1,:,:],2, dim=(-2,-1))
    XY = torch.exp(-XY * XY / (2 * sigma * sigma)).mean()
    print('cross: ', XY.item())

    return XX + YY - 2 * XY

def load_point_cloud_xyz(file):
    point_cloud = np.fromfile(file, dtype=np.float32).reshape(-1, 4)[:,:3]
    depth = np.linalg.norm(point_cloud[:,:3], 2, axis=1)
    mask = np.logical_and(depth>3.0, depth < 70.0)
    point_cloud = point_cloud[mask,:]
    return point_cloud

def load_point_cloud_xyz_nus(file, to_range=None):
    if to_range:
        pth_path = file.replace('sweeps', 'sweeps_range').replace('.bin', '.pth')
        pts_range = torch.load(pth_path)['jpg'].unsqueeze(0)
        point_cloud = to_range.to_pc_torch(pts_range)[0].cpu().detach().numpy()
    else:
        point_cloud = np.fromfile(file, dtype=np.float32).reshape(-1, 5)[:,:3]
    depth = np.linalg.norm(point_cloud[:,:3], 2, axis=1)
    mask = np.logical_and(depth>2.0, depth < 90.0)
    point_cloud = point_cloud[mask,:]
    return point_cloud

def calculate_mmd_nus(sample_folder, from_bin=True):
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
    
    nus_model_distance = compute_mmd(nus_histograms, model_histograms, gaussian, is_hist=True)

    return nus_model_distance

def calculate_mmd(sample_folder, from_bin=False):
    if from_bin:
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

    kitti_model_distance = compute_mmd(kitti_histograms, model_histograms, gaussian, is_hist=True)

    return kitti_model_distance