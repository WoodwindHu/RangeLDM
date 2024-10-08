from collections import defaultdict
from glob import glob
import json
import os
import random
from typing import Any
from dataset import RangeDataset, RangeLoader, point_cloud_to_range_image
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
import torch
from typing import Optional, Tuple
        

class point_cloud_to_range_image_nuScenes(point_cloud_to_range_image):
    def __init__(self, 
                **kwargs,) -> None:
        super().__init__(**kwargs)
        self.height = np.array([-0.00216031, -0.00098729, -0.00020528,  0.00174976,  0.0044868 , -0.00294233,
                                -0.00059629, -0.00020528,  0.00174976, -0.00294233, -0.0013783 ,  0.00018573,
                                 0.00253177, -0.00098729,  0.00018573,  0.00096774, -0.00411535, -0.0013783,
                                 0.00018573,  0.00018573, -0.00294233, -0.0013783 , -0.00098729, -0.00020528,
                                 0.00018573,  0.00018573,  0.00018573, -0.00020528,  0.00018573,  0.00018573,
                                 0.00018573,  0.00018573,], dtype=np.float32)
        self.zenith = np.array([ 1.86705767e-01,  1.63245357e-01,  1.39784946e-01,  1.16324536e-01,
                                 9.28641251e-02,  7.01857283e-02,  4.67253177e-02,  2.32649071e-02,
                                -1.95503421e-04, -2.28739003e-02, -4.63343109e-02, -6.97947214e-02,
                                -9.32551320e-02, -1.15933529e-01, -1.39393939e-01, -1.62854350e-01,
                                -1.85532747e-01, -2.08993157e-01, -2.32453568e-01, -2.55913978e-01,
                                -2.78592375e-01, -3.02052786e-01, -3.25513196e-01, -3.48973607e-01,
                                -3.72434018e-01, -3.95894428e-01, -4.19354839e-01, -4.42033236e-01,
                                -4.65493646e-01, -4.88954057e-01, -5.12414467e-01, -5.35874878e-01,], dtype=np.float32)
        self.incl = -self.zenith
        self.H = 32

    def __call__(self, pc) -> Any:
        depth = np.linalg.norm(pc[:,:3], 2, axis=1)
        mask = depth > 2.0
        pc = pc[mask, :]
        return super().__call__(pc)

    def get_row_inds(self, pc):
        row_inds = 31 - pc[:, 4].astype(np.int32) # nuscenes already has the row_inds
        return row_inds


class nuScenesRangeDataset(RangeDataset):
    def __init__(self, 
                 nuScenes_path, 
                 train=True, 
                 width=1024, 
                 grid_sizes=[1, 1024, 1024, ], 
                 pc_range=[-25.6, -25.6, -3., 25.6, 25.6, 1.], 
                 log=False,
                 inverse=False,
                 **kwargs):
        super().__init__(**kwargs)
        if train:
            with open(os.path.join(nuScenes_path, 'v1.0-trainval/sample_data.json')) as f:
                sample_data = json.load(f)
        else:
            with open(os.path.join(nuScenes_path, 'v1.0-test/sample_data.json')) as f:
                sample_data = json.load(f)

        file_paths = [os.path.join(nuScenes_path, x['filename']) 
                           for x in sample_data 
                           if 'sweeps/LIDAR_TOP' in x['filename']]
        self.file_paths = sorted(file_paths)
        self.to_range_image = point_cloud_to_range_image_nuScenes(width=width, 
                                                               grid_sizes=grid_sizes, 
                                                               pc_range=pc_range, 
                                                               log=log,
                                                               inverse=inverse)

    def get_pts(self, pts_path):
        pts = np.fromfile(pts_path, dtype=np.float32).reshape(-1, 5)
        pts[:, 3] = pts[:, 3] / 255.0
        return pts

    def get_pth_path(self, pts_path):
        return pts_path.replace('sweeps', 'sweeps_range').replace('.bin', '.pth')

class nuScenesRangeLoader(RangeLoader):
    def __init__(self, 
                 nuScenes_path, 
                 used_feature=2, 
                 width=1024, 
                 grid_sizes=[1, 1024, 1024, ], 
                 pc_range=[-25.6, -25.6, -3., 25.6, 25.6, 1.], 
                 log=False, 
                 inverse=False, 
                 downsample=None,
                 inpainting=None,
                 coord=False,
                 **kwargs):
        super().__init__(**kwargs)
        dataset = nuScenesRangeDataset(nuScenes_path, 
                                    train=True, 
                                    used_feature=used_feature, 
                                    width=width, 
                                    grid_sizes=grid_sizes, 
                                    pc_range=pc_range, 
                                    log=log,
                                    inverse=inverse,
                                    downsample=downsample,
                                    inpainting=inpainting,
                                    coord=coord)
        test_dataset = nuScenesRangeDataset(nuScenes_path, 
                                         train=False, 
                                         used_feature=used_feature, 
                                         width=width, 
                                         grid_sizes=grid_sizes, 
                                         pc_range=pc_range, 
                                         log=log,
                                         inverse=inverse,
                                         downsample=downsample,
                                         inpainting=inpainting,
                                         coord=coord)
        self.train_dataset = dataset
        self.test_dataset = test_dataset

if __name__ == '__main__':
    from PIL import Image
    from tqdm import tqdm
    batch_size=32
    loader = nuScenesRangeLoader(os.environ.get('NUSCENES_DATASET'), 
                                        batch_size=batch_size, 
                                        num_workers=1, 
                                        )
    to_range = loader.test_dataset.to_range_image
    to_range.mean = 50.
    to_range.std = 50.
    out_path = 'test_range_image_nuscenes'
    os.makedirs(out_path, exist_ok=True)
    # for i in tqdm(loader.test_dataloader()):
    #     pass
    dataloader_iter = iter(loader.test_dataloader())
    for i in tqdm(range(1000//batch_size + 1)):
        batch=next(dataloader_iter)
        pc_in_all = to_range.to_pc_torch(batch['jpg'])
        bev_in_all = to_range.to_voxel(batch['jpg'])
        for j in range(batch_size):
            if i*batch_size + j >= 1000:
                break
            pc = pc_in_all[j].cpu().detach().numpy()
            depth = np.linalg.norm(pc[:,:3], 2, axis=1)
            mask = depth < 90.0
            pc[mask, :].tofile(f'{out_path}/in{i * batch_size+j}.bin')
            # bev_image = Image.fromarray((bev_in_all[j].permute(2, 1, 0).cpu().detach().numpy().clip(0, 1)*255.).astype(np.uint8)[:,:,0], mode='L')
            # bev_image.save(f'{out_path}/in{j}.png')