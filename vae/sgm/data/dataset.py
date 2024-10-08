from collections import defaultdict
from glob import glob
import os
import random
from typing import Any
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
import torch
from typing import Optional, Tuple

def _splat_points_to_volumes(
    points_3d: torch.Tensor,
    points_features: torch.Tensor,
    volume_densities: torch.Tensor,
    volume_features: torch.Tensor,
    grid_sizes: torch.LongTensor,
    min_weight: float = 1e-4,
    mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convert a batch of point clouds to a batch of volumes using trilinear
    splatting into a volume.

    Args:
        points_3d: Batch of 3D point cloud coordinates of shape
            `(minibatch, N, 3)` where N is the number of points
            in each point cloud. Coordinates have to be specified in the
            local volume coordinates (ranging in [-1, 1]).
        points_features: Features of shape `(minibatch, N, feature_dim)`
            corresponding to the points of the input point cloud `points_3d`.
        volume_features: Batch of input *flattened* feature volumes
            of shape `(minibatch, feature_dim, N_voxels)`
        volume_densities: Batch of input *flattened* feature volume densities
            of shape `(minibatch, N_voxels, 1)`. Each voxel should
            contain a non-negative number corresponding to its
            opaqueness (the higher, the less transparent).
        grid_sizes: `LongTensor` of shape (3) representing the
            spatial resolutions of each of the the non-flattened `volumes` tensors.
            Note that the following has to hold:
                `torch.prod(grid_sizes)==N_voxels`
        min_weight: A scalar controlling the lowest possible total per-voxel
            weight used to normalize the features accumulated in a voxel.
        mask: A binary mask of shape `(minibatch, N)` determining which 3D points
            are going to be converted to the resulting volume.
            Set to `None` if all points are valid.
    Returns:
        volume_features: Output volume of shape `(minibatch, D, N_voxels)`.
        volume_densities: Occupancy volume of shape `(minibatch, 1, N_voxels)`
            containing the total amount of votes cast to each of the voxels.
    """

    _, n_voxels, density_dim = volume_densities.shape
    ba, n_points, feature_dim = points_features.shape

    # minibatch x n_points x feature_dim -> minibatch x feature_dim x n_points
    points_features = points_features.permute(0, 2, 1).contiguous()

    # XYZ = the upper-left volume index of the 8-neighborhood of every point
    # grid_sizes is of the form (minibatch, depth-height-width)
    grid_sizes_xyz = grid_sizes[None, [2, 1, 0]]

    # Convert from points_3d in the range [-1, 1] to
    # indices in the volume grid in the range [0, grid_sizes_xyz-1]
    points_3d_indices = ((points_3d + 1) * 0.5) * (
        grid_sizes_xyz[:, None].type_as(points_3d) - 1
    )
    XYZ = points_3d_indices.floor().long()
    rXYZ = points_3d_indices - XYZ.type_as(points_3d)  # remainder of floor

    # split into separate coordinate vectors
    X, Y, Z = XYZ.split(1, dim=2)
    # rX = remainder after floor = 1-"the weight of each vote into
    #      the X coordinate of the 8-neighborhood"
    rX, rY, rZ = rXYZ.split(1, dim=2)

    # get random indices for the purpose of adding out-of-bounds values
    rand_idx = X.new_zeros(X.shape).random_(0, n_voxels)

    # iterate over the x, y, z indices of the 8-neighborhood (xdiff, ydiff, zdiff)
    for xdiff in (0, 1):
        X_ = X + xdiff
        wX = (1 - xdiff) + (2 * xdiff - 1) * rX
        for ydiff in (0, 1):
            Y_ = Y + ydiff
            wY = (1 - ydiff) + (2 * ydiff - 1) * rY
            for zdiff in (0, 1):
                Z_ = Z + zdiff
                wZ = (1 - zdiff) + (2 * zdiff - 1) * rZ

                # weight of each vote into the given cell of 8-neighborhood
                w = wX * wY * wZ

                # valid - binary indicators of votes that fall into the volume
                valid = (
                    (0 <= X_)
                    * (X_ < grid_sizes_xyz[:, None, 0:1])
                    * (0 <= Y_)
                    * (Y_ < grid_sizes_xyz[:, None, 1:2])
                    * (0 <= Z_)
                    * (Z_ < grid_sizes_xyz[:, None, 2:3])
                ).long()

                # linearized indices into the volume
                idx = (Z_ * grid_sizes[None, None, 1:2] + Y_) * grid_sizes[
                    None, None, 2:3
                ] + X_

                # out-of-bounds features added to a random voxel idx with weight=0.
                idx_valid = idx * valid + rand_idx * (1 - valid)
                w_valid = w * valid.type_as(w)
                if mask is not None:
                    w_valid = w_valid * mask.type_as(w)[:, :, None]

                # scatter add casts the votes into the weight accumulator
                # and the feature accumulator
                volume_densities.scatter_add_(1, idx_valid, w_valid)

                # reshape idx_valid -> (minibatch, feature_dim, n_points)
                idx_valid = idx_valid.view(ba, 1, n_points).expand_as(points_features)
                w_valid = w_valid.view(ba, 1, n_points)

                # volume_features of shape (minibatch, feature_dim, n_voxels)
                volume_features.scatter_add_(2, idx_valid, w_valid * points_features)

    # divide each feature by the total weight of the votes
    volume_features = volume_features / volume_densities.view(ba, 1, n_voxels).clamp(
        min_weight
    )

    return volume_features, volume_densities
        

class point_cloud_to_range_image:
    def __init__(self, 
                width=1024, 
                grid_sizes=[1, 1024, 1024, ], 
                pc_range=[-25.6, -25.6, -3., 25.6, 25.6, 1.], 
                log=False, 
                normalize_volume_densities=True,
                inverse=False,) -> None:
        self.height = None
        self.zenith = None
        self.incl = None
        self.range_fill_value = np.array([100, 0])
        self.width = width
        self.grid_sizes = grid_sizes
        self.pc_range = pc_range
        self.log = log
        self.normalize_volume_densities = normalize_volume_densities
        self.inverse = inverse
        self.mean = 20.
        self.std = 40.
    
    def get_row_inds(self, pc):
        raise NotImplementedError

    def __call__(self, pc) -> Any:
        row_inds = self.get_row_inds(pc)

        azi = np.arctan2(pc[:,1], pc[:,0])
        col_inds = self.width - 1.0 + 0.5 - (azi + np.pi) / (2.0 * np.pi) * self.width
        col_inds = np.round(col_inds).astype(np.int32)
        col_inds[col_inds == self.width] = self.width - 1
        col_inds[col_inds < 0] = 0
        empty_range_image = np.full((self.H, self.width, 2), -1, dtype = np.float32)
        pc[:,2] -= self.height[row_inds]
        point_range = np.linalg.norm(pc[:,:3], axis = 1, ord = 2)
        point_range[point_range > self.range_fill_value[0]] = self.range_fill_value[0]

        order = np.argsort(-point_range)
        if self.log:
            point_range = np.log2(point_range[order]+1)/6
        elif self.inverse:
            point_range = 1/point_range[order]
        else:
            point_range = point_range[order] 
        pc = pc[order]
        row_inds = row_inds[order]
        col_inds = col_inds[order]

        empty_range_image[row_inds, col_inds, :] = np.concatenate([point_range[:,None], pc[:,3:4]], axis = 1)

        return empty_range_image
    
    @staticmethod
    def fill_noise(data, miss_inds, width):
        data_shift1pxl = data[:, list(range(1, width)) + [0, ], :]
        data[miss_inds, :] = data_shift1pxl[miss_inds, :]
        return data
    
    def process_miss_value(self, range_image):
        range_image_mask = range_image[..., 0] > 0
        height, width, _ = range_image.shape
        miss_inds = range_image[:, :, 0] == -1

        range_image = self.fill_noise(range_image, miss_inds, width)
        range_image_mask = self.fill_noise(range_image_mask[:, :, None], miss_inds, width).squeeze()

        still_miss_inds = range_image[:, :, 0] == -1

        shift_down_2px = range_image[[height - 2, height - 1] + list(range(height - 2)), :, 0]
        shift_top_2px = range_image[list(range(2, height)) + [0, 1], :, 0]
        shift_right_2px = range_image[:, [width - 2, width - 1] + list(range(width - 2)), 0]
        shift_left_2px = range_image[:, list(range(2, width)) + [0, 1], 0]

        car_window_mask = still_miss_inds & ((shift_down_2px != -1) | (shift_top_2px != -1) |
                                             (shift_right_2px != -1) | (shift_left_2px != -1))

        if self.log:
            range_image[still_miss_inds, :] = np.log2(self.range_fill_value+1)/6
        elif self.inverse:
            range_image[still_miss_inds, :] = np.array([1/self.range_fill_value[0], self.range_fill_value[1]])
        else:
            range_image[still_miss_inds, :] = self.range_fill_value 

        # How much are the intensity and elongation of car windows
        # range_image[car_window_mask, :] = np.array([0, 0])

        return range_image, range_image_mask, car_window_mask

    def normalize(self, range_image):
        if not self.log and not self.inverse:
            range_image[..., 0] = (range_image[..., 0] - self.mean) / self.std
        return range_image
    
    def to_pc_torch(self, range_images):
        '''
        range_images: Bx2xWxH
        output:
            point_cloud: BxNx4
        '''
        device = range_images.device
        incl_t = torch.from_numpy(self.incl).to(device)
        height_t = torch.from_numpy(self.height).to(device)
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

        # Calculate theta
        theta = torch.pi / 2 - incl_t


        r_true = point_range 

        r_true[r_true<0] = self.range_fill_value[0]

        # Calculate z
        z = (height_t[None,None,:] - r_true * torch.sin(incl_t[None,None,:])).reshape(batch_size, -1)

        # Calculate xy_norm
        xy_norm = r_true * torch.cos(incl_t[None,None,:])

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

    def to_voxel(self, range_images):
        batch_size = range_images.shape[0]
        pc = self.to_pc_torch(range_images)
        grid_sizes = torch.LongTensor(self.grid_sizes).to(pc.device)
        pc_range = torch.Tensor(self.pc_range).to(pc.device)
        pc[:,:,:3] -= (pc_range[None, None, 3:] + pc_range[None, None, :3])/2
        pc[:,:,:3] /= (pc_range[None, None, 3:] - pc_range[None, None, :3])/2
        volume_densities = torch.zeros(batch_size, np.prod(self.grid_sizes), 1).to(pc.device)
        volume_features = torch.zeros(batch_size, 1, np.prod(self.grid_sizes)).to(pc.device)
        volume_features, volume_densities = _splat_points_to_volumes(pc[:,:,:3], pc[:,:,3:], volume_densities, volume_features, grid_sizes)
        if self.normalize_volume_densities:
            volume_densities = torch.log(volume_densities+1) # normalize volume_densities
        volume_densities = volume_densities.view(batch_size, *self.grid_sizes)
        volume_features = volume_features.view(batch_size, *self.grid_sizes)
        voxel = torch.cat([volume_densities, volume_features], dim=1)
        return voxel



class RangeDataset(Dataset):
    def __init__(self, 
                 used_feature=2, 
                 downsample=None,
                 inpainting=None,
                 coord=False):
        self.file_paths = None
        self.to_range_image = None
        self.used_feature = used_feature
        self.downsample = downsample
        self.coord = coord
        self.inpainting = inpainting

    def get_pts(self, pts_path):
        raise NotImplementedError
    
    def get_pth_path(self, pts_path):
        raise NotImplementedError

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        pts_path = self.file_paths[idx]
        pth_path = self.get_pth_path(pts_path)
        if os.path.exists(pth_path):
            ret = torch.load(pth_path)
        else:
            pts = self.get_pts(pts_path)
            range_image = self.to_range_image(pts)
            range_image, range_image_mask, car_window_mask  = self.to_range_image.process_miss_value(range_image)
            range_image = self.to_range_image.normalize(range_image)
            range_image = torch.from_numpy(range_image).permute(2, 1, 0) # (C, W, H)
            range_image_mask = torch.from_numpy(range_image_mask).permute(1, 0)
            car_window_mask = torch.from_numpy(car_window_mask).permute(1, 0)
            range_image = range_image[:self.used_feature]
            if self.coord:
                coord = (torch.arange(range_image.shape[2], dtype=torch.float32)/range_image.shape[2]).reshape(1, 1, -1).repeat(1, range_image.shape[1], 1)
                range_image = torch.cat([range_image, coord], dim=0)
            ret = {'jpg': range_image, 
                    'mask': range_image_mask, 
                    'car_window_mask': car_window_mask}
        if self.downsample:
            if isinstance(self.downsample, int):
                self.downsample = [1, self.downsample]
            range_image = ret['jpg']
            ret.update({'down': range_image[:, 
                                            (self.downsample[0]//2)::self.downsample[0], 
                                            (self.downsample[1]//2)::self.downsample[1]]})
        if self.inpainting:
            # draw mask image
            range_image = ret['jpg']
            C, W, H = range_image.shape
            inpainting_mask = -torch.ones(1,W,H)
            inpainting_start = 0. # random.random()
            inpainting_end = inpainting_start+self.inpainting
            if inpainting_end < 1.:
                inpainting_mask[:, int(inpainting_start*W):int(inpainting_end*W),:] = 1
            else:
                inpainting_mask[:, int(inpainting_start*W):,:] = 1
                inpainting_mask[:, :int((inpainting_end-1.)*W),:] = 1
            masked_image = -torch.ones_like(range_image)
            masked_image[:, inpainting_mask.squeeze(0) < 0] = range_image[:, inpainting_mask.squeeze(0) < 0]
            ret.update({'inpainting_mask': inpainting_mask,
                        'masked_image': masked_image}) 
        if not os.path.exists(pth_path):
            pth_path = Path(pth_path)
            os.makedirs(str(pth_path.parent), exist_ok=True)
            torch.save(ret, pth_path)
        return ret


    @staticmethod
    def collate_batch(batch_list, _unused=False):
        data_dict = defaultdict(list)
        for cur_sample in batch_list:
            for key, val in cur_sample.items():
                data_dict[key].append(val)
        batch_size = len(batch_list)
        ret = {}
        for key, val in data_dict.items():
            ret[key] = torch.stack([imgs for imgs in val],dim=0)
        return ret

class RangeLoader(pl.LightningDataModule):
    def __init__(self, 
                 batch_size, 
                 num_workers=0, 
                 shuffle=True,):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle

    def prepare_data(self):
        pass

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )
