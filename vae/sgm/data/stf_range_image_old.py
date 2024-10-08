import numpy as np
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
import torch


def point_cloud_to_range_image_STF(point_cloud, log_map=False, H: int=64, W: int=1024, max_depth=100.):
    depth = np.linalg.norm(point_cloud[:,:3], 2, axis=1)
    depth_mask = depth<max_depth
    depth = depth[depth_mask]
    point_cloud = point_cloud[depth_mask]
    ring = 63 - point_cloud[:,4].astype(np.int32)
    points = point_cloud[:,:3]
    remission = point_cloud[:,3]
    order = np.argsort(depth)[::-1]
    scan_x = points[:, 0]
    scan_y = points[:, 1]
    scan_z = points[:, 2]
    pitch = np.arcsin(scan_z / depth)
    yaw = -np.arctan2(scan_y, scan_x)
    proj_x = 0.5 * (yaw / np.pi + 1.0)          # in [0.0, 1.0]
    proj_x *= float(W)                              # in [0.0, W]
    # round and clamp for use as index
    proj_x = np.floor(proj_x)
    proj_x = np.minimum(W - 1, proj_x)
    proj_x = np.maximum(0, proj_x).astype(np.int32)   # in [0,W-1]
    # projected range image - [H,W] range (0 is no data)
    proj_range = np.full((H, W), 0, dtype=np.float32)
    # projected remission - [H,W] intensity (0 is no data)
    proj_remission = np.full((H, W), 0, dtype=np.float32)
    proj_pitch = np.full((H, W), 0, dtype=np.float32)
    depth = depth[order]
    remission = remission[order]
    ring = ring[order]
    proj_x = proj_x[order]
    pitch = pitch[order]
    if log_map:
        depth = np.log2(depth+1)/6
    else:
        depth = depth / max_depth
    proj_range[ring, proj_x] = depth
    proj_remission[ring, proj_x] = remission
    proj_pitch[ring, proj_x] = pitch + 0.5
    image = np.concatenate((proj_range[:,:,None], proj_pitch[:,:,None], proj_remission[:,:,None]), axis=2)
    return image

def range_image_to_point_cloud_stf(image, log_map=False, H = 64.0, W = 1024.0, max_depth=100.):
    # breakpoint()
    lidar_range = image[:,:,0] # range
    if log_map:
        depth_range = np.exp2(lidar_range*6)-1
    else:
        depth_range = lidar_range * max_depth
    depth = depth_range.flatten()
    lidar_pitch = image[:,:,1]
    pitch = lidar_pitch.flatten() - 0.5
    lidar_intensity = image[:,:,2] # intensity
    lidar_intensity = lidar_intensity.flatten()
    
    
    x, y = np.meshgrid(np.arange(0, W), np.arange(0, H))
    x += 0.5
    x *= 1/W
    y *= 1/H
    yaw = np.pi*(x * 2 - 1)
    yaw = yaw.flatten()
    pts = np.zeros((len(yaw), 3))
    pts[:, 0] =  np.cos(yaw) * np.cos(pitch) * depth
    pts[:, 1] =  -np.sin(yaw) * np.cos(pitch) * depth
    pts[:, 2] =  np.sin(pitch) * depth

    # mask out invalid points
    mask = np.logical_and(depth>0.5, depth < max_depth)
    xyz = pts[mask, :]
    lidar_intensity = lidar_intensity[mask, None]
    lidar = np.concatenate((xyz, lidar_intensity), axis=1, dtype=np.float32)
    return lidar


def range_image_to_point_cloud_stf_torch(image, log_map=False, H=64.0, W=1024.0, max_depth=100.):
    '''
    image: Bx3xWxH
    '''
    batch_size = image.shape[0]
    lidar_range = image[:, 0]  # range
    if log_map:
        depth_range = torch.exp2(lidar_range * 6) - 1
    else:
        depth_range = lidar_range * max_depth
    depth = depth_range.view(batch_size, -1)
    lidar_pitch = image[:, 1]
    pitch = lidar_pitch.view(batch_size, -1) - 0.5
    lidar_intensity = image[:, 2]  # intensity
    lidar_intensity = lidar_intensity.view(batch_size, -1)

    x, y = torch.meshgrid(torch.arange(0, W, device=image.device), torch.arange(0, H, device=image.device))
    x = (x.clone() + 0.5)/ W
    y = y.clone() / H
    yaw = torch.tensor(3.141592653589793, device=image.device) * (x * 2 - 1)
    yaw = yaw.view(-1)
    pts = torch.zeros( (batch_size, len(yaw), 3), dtype=torch.float32, device=image.device)
    pts[:, :, 0] = torch.cos(yaw)[None, :] * torch.cos(pitch) * depth
    pts[:, :, 1] = -torch.sin(yaw)[None, :] * torch.cos(pitch) * depth
    pts[:, :, 2] = torch.sin(pitch) * depth

    lidar_intensity = lidar_intensity.unsqueeze(2)
    lidar = torch.cat((pts, lidar_intensity), dim=2)
    return lidar

class STFRangeDataset(Dataset):
    def __init__(self, STF_path, split='dense_fog_day', lidar='lidar_hdl64_strongest', max_depth=100.):
        STF_path = Path(STF_path) 
        split_file = STF_path / 'splits' / f'{split}.txt'
        with open(split_file, 'r') as infile:
            indexes_split = infile.readlines()
        indexes_split = [i.replace(',','_').strip() for i in indexes_split]
        indexes_split.sort()
        self.file_paths = [STF_path / lidar / f'{file}.bin' for file in indexes_split]
        self.max_depth = max_depth

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        pts_path = self.file_paths[idx]
        pts = np.fromfile(pts_path, dtype=np.float32).reshape(-1, 5)
        pts[:, 3] = pts[:,3] / 255.
        range_image = point_cloud_to_range_image_STF(pts, log_map=True, max_depth=self.max_depth)
        range_image = torch.from_numpy(range_image).permute(2, 1, 0)
        return {'jpg': range_image}

    @staticmethod
    def collate_batch(batch_list, _unused=False):
        batch_size = len(batch_list)
        shape = batch_list[0]['jpg'].shape
        ret = torch.zeros((batch_size, shape[0], shape[1], shape[2]))
        for i in range(batch_size):
            ret[i] = batch_list[i]['jpg']
        return {'jpg': ret}

class STFRangeLoader(pl.LightningDataModule):
    def __init__(self, STF_path, batch_size, num_workers=0, shuffle=True, max_depth=100.):
        super().__init__()
        dataset = STFRangeDataset(STF_path, 'train_clear', max_depth=max_depth)
        test_dataset = STFRangeDataset(STF_path, 'test_clear', max_depth=max_depth)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.train_dataset = dataset
        self.test_dataset = test_dataset

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
            shuffle=self.shuffle,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
        )
