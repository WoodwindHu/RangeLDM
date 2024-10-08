from collections import defaultdict
from glob import glob
import os
import random
from typing import Any
from .dataset import RangeDataset, RangeLoader, point_cloud_to_range_image
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
import torch
from typing import Optional, Tuple
        

class point_cloud_to_range_image_KITTI_vanilla(point_cloud_to_range_image):
    def __init__(self, 
                **kwargs,) -> None:
        super().__init__(**kwargs)
        self.H = 64
        self.proj_fov_up = 3.0 / 180.0 * np.pi
        self.proj_fov_down = -25.0 / 180.0 * np.pi
        self.fov = self.proj_fov_up - self.proj_fov_down
        self.height = np.zeros(64)

    def get_row_inds(self, pc):
        point_range = np.linalg.norm(pc[:,:3], axis = 1, ord = 2)
        zen = np.arcsin(pc[:,2] / point_range)

        row_inds = 64.0  - 1.0 + 0.5 - (zen - self.proj_fov_down) / self.fov * 64.0
        row_inds = np.round(row_inds).astype(np.int32)
        row_inds[row_inds == 64] = 63
        row_inds[row_inds < 0] = 0
        return row_inds
    
    def to_pc_torch(self, range_images):
        '''
        range_images: Bx2xWxH
        output:
            point_cloud: BxNx4
        '''
        device = range_images.device
        batch_size, channels, width_dim, height_dim = range_images.shape

        # Extract point range and remission
        if self.log:
            point_range = 2**(range_images[:, 0, :, :] * 6) - 1
        elif self.inverse:
            point_range = 1/torch.max(range_images[:, 0, :, :], torch.Tensor([0.0001]).to(device))
        else:
            point_range = range_images[:, 0, :, :] * self.std + self.mean # BxWxH
        if range_images.shape[1] > 1:
            remission = range_images[:, 1, :, :].reshape(batch_size, -1)


        r_true = point_range 

        # Calculate zen
        height = height_dim
        zen = (height - 0.5 - torch.arange(0, height, device=device)) / height * self.fov + self.proj_fov_down

        # Calculate z
        z = (r_true * torch.sin(zen[None,None,:])).reshape(batch_size, -1)
        

        # Calculate xy_norm
        xy_norm = r_true * torch.cos(zen[None,None,:])

        # Calculate azi
        width = width_dim
        azi = (width - 0.5 - torch.arange(0, width, device=device)) / width * 2. * torch.pi - torch.pi

        # Calculate x and y
        x = (xy_norm * torch.cos(azi[None,:,None])).reshape(batch_size, -1)
        y = (xy_norm * torch.sin(azi[None,:,None])).reshape(batch_size, -1)

        # Concatenate the arrays to create the point cloud
        if range_images.shape[1] > 1:
            point_cloud = torch.stack([x, y, z, remission], dim=2)
        else:
            point_cloud = torch.stack([x, y, z], dim=2)

        return point_cloud



class KITTIRangeVanillaDataset(RangeDataset):
    def __init__(self, 
                 KITTI_path, 
                 train=True, 
                 width=1024, 
                 grid_sizes=[1, 1024, 1024, ], 
                 pc_range=[-25.6, -25.6, -3., 25.6, 25.6, 1.], 
                 log=False,
                 inverse=False,
                 **kwargs):
        super().__init__(**kwargs)
        full_list = glob(os.path.join(KITTI_path, 'data_3d_raw/*/velodyne_points/data/*.bin'))
        if train:
            full_list = sorted(list(filter(lambda file: '0000_sync' not in file and '0002_sync' not in file, full_list)))
        else:
            full_list = sorted(list(filter(lambda file: '0000_sync' in file or '0002_sync' in file, full_list)))
        self.file_paths = full_list
        self.to_range_image = point_cloud_to_range_image_KITTI_vanilla(width=width, 
                                                               grid_sizes=grid_sizes, 
                                                               pc_range=pc_range, 
                                                               log=log,
                                                               inverse=inverse)

    def get_pts(self, pts_path):
        return np.fromfile(pts_path, dtype=np.float32).reshape(-1, 4)
    
    def get_pth_path(self, pts_path):
        return pts_path.replace('data_3d_raw', 'data_3d_range_vanilla').replace('.bin', '.pth')




class KITTIRangeVanillaLoader(RangeLoader):
    def __init__(self, 
                 KITTI_path, 
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
        dataset = KITTIRangeVanillaDataset(KITTI_path, 
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
        test_dataset = KITTIRangeVanillaDataset(KITTI_path, 
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
    batch_size=16
    loader = KITTIRangeVanillaLoader(os.environ.get('KITTI360_DATASET'), 
                                        batch_size=batch_size, 
                                        num_workers=8, 
                                        )
    to_range = loader.test_dataset.to_range_image
    out_path = 'test_range_image_vanilla'
    os.makedirs(out_path, exist_ok=True)
    for batch in loader.test_dataloader():
        pc_in_all = to_range.to_pc_torch(batch['jpg'])
        bev_in_all = to_range.to_voxel(batch['jpg'])
        for j in range(batch_size):
            pc = pc_in_all[j].cpu().detach().numpy()
            depth = np.linalg.norm(pc[:,:3], 2, axis=1)
            mask = depth < 70.0
            pc[mask, :].tofile(f'{out_path}/in{j}.bin')
            bev_image = Image.fromarray((bev_in_all[j].permute(2, 1, 0).cpu().detach().numpy().clip(0, 1)*255.).astype(np.uint8)[:,:,0], mode='L')
            bev_image.save(f'{out_path}/in{j}.png')
        break