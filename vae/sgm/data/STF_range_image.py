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
        

class point_cloud_to_range_image_STF(point_cloud_to_range_image):
    def __init__(self, 
                **kwargs,) -> None:
        super().__init__(**kwargs)
        self.height = np.array(
            [0.20428571, 0.20534247, 0.20551859, 0.20587084, 0.20587084,
            0.20604697, 0.20675147, 0.20745597, 0.20763209, 0.20710372,
            0.20727984, 0.2090411 , 0.20956947, 0.20921722, 0.21080235,
            0.20992172, 0.21027397, 0.20921722, 0.21238748, 0.21273973,
            0.21414873, 0.21379648, 0.21520548, 0.21168297, 0.2153816 ,
            0.21749511, 0.22101761, 0.21432485, 0.22101761, 0.21626223,
            0.21714286, 0.21908023, 0.14510763, 0.1435225 , 0.14845401,
            0.14827789, 0.14863014, 0.14933464, 0.14898239, 0.15303327,
            0.15320939, 0.15320939, 0.15514677, 0.15655577, 0.15426614,
            0.15690802, 0.15585127, 0.15902153, 0.15990215, 0.16131115,
            0.16078278, 0.16448141, 0.16395303, 0.16712329, 0.16694716,
            0.16958904, 0.17046967, 0.17293542, 0.17240705, 0.17434442,
            0.1741683 , 0.17786693, 0.17857143, 0.18103718    ], dtype=np.float32)
        self.zenith = np.array([
            0.03336595,  0.02749511,  0.02162427,  0.01575342,  0.00890411,
            0.00401174, -0.0018591 , -0.00870841, -0.01360078, -0.01947162,
            -0.02632094, -0.03219178, -0.03806262, -0.04295499, -0.04980431,
            -0.05469667, -0.06154599, -0.06741683, -0.07426614, -0.07915851,
            -0.08502935, -0.0909002 , -0.09774951, -0.10264188, -0.10949119,
            -0.11634051, -0.12221135, -0.12612524, -0.13297456, -0.1388454 ,
            -0.14471624, -0.14863014, -0.15450098, -0.16428571, -0.1721135 ,
            -0.17994129, -0.18874755, -0.19951076, -0.20831703, -0.21908023,
            -0.22592955, -0.23473581, -0.24158513, -0.25430528, -0.26213307,
            -0.27191781, -0.27876712, -0.28757339, -0.29540117, -0.30812133,
            -0.31692759, -0.3276908 , -0.3316047 , -0.34334638, -0.35019569,
            -0.36193738, -0.37074364, -0.38150685, -0.38835616, -0.39618395,
            -0.40401174, -0.4167319 , -0.42455969, -0.43434442
        ], dtype=np.float32) 
        self.incl = -self.zenith
        self.H = 64

    def get_row_inds(self, pc):
        row_inds = 63 - pc[:, 4].astype(np.int32) # nuscenes already has the row_inds
        return row_inds



class STFRangeDataset(RangeDataset):
    def __init__(self, 
                 STF_path, 
                 train=True, 
                 width=1024, 
                 grid_sizes=[1, 1024, 1024, ], 
                 pc_range=[-25.6, -25.6, -3., 25.6, 25.6, 1.], 
                 log=False,
                 inverse=False,
                 dataset_cfg=None,
                 **kwargs):
        super().__init__(**kwargs)
        self.root_path = Path(STF_path)
        self.split = dataset_cfg.split
        self.root_split_path = self.root_path

        self.sensor_type = dataset_cfg.SENSOR_TYPE
        self.signal_type = dataset_cfg.SIGNAL_TYPE

        self.suffix = '_vlp32' if self.sensor_type == 'vlp32' else ''

        split_dir = self.root_path / 'ImageSets' / f'{self.split}{self.suffix}.txt'

        print(split_dir)
        if split_dir.exists():
            self.file_paths = [os.path.join(self.root_path, 
                                            f'lidar_{self.sensor_type}_{self.signal_type}',
                                            x.strip().replace(',', '_')+'.bin') 
                                for x in open(split_dir).readlines()]
        self.to_range_image = point_cloud_to_range_image_STF(width=width, 
                                                               grid_sizes=grid_sizes, 
                                                               pc_range=pc_range, 
                                                               log=log,
                                                               inverse=inverse)
    def get_pts(self, pts_path):
        pts =  np.fromfile(pts_path, dtype=np.float32).reshape(-1, 5)
        pts[:, 3] = pts[:, 3] / 255.0
        return pts
    
    def get_pth_path(self, pts_path):
        return pts_path.replace(f'lidar_{self.sensor_type}_{self.signal_type}', 
                            f'lidar_range_{self.sensor_type}_{self.signal_type}').replace('.bin', '.pth')

class STFRangeLoader(RangeLoader):
    def __init__(self, 
                 STF_path, 
                 used_feature=2, 
                 width=1024, 
                 grid_sizes=[1, 1024, 1024, ], 
                 pc_range=[-25.6, -25.6, -3., 25.6, 25.6, 1.], 
                 log=False, 
                 inverse=False, 
                 downsample=None,
                 inpainting=None,
                 coord=False,
                 dataset_cfg=None,
                 **kwargs):
        super().__init__(**kwargs)
        dataset = STFRangeDataset(STF_path, 
                                    train=True, 
                                    used_feature=used_feature, 
                                    width=width, 
                                    grid_sizes=grid_sizes, 
                                    pc_range=pc_range, 
                                    log=log,
                                    inverse=inverse,
                                    downsample=downsample,
                                    inpainting=inpainting,
                                    coord=coord,
                                    dataset_cfg=dataset_cfg)
        test_dataset = STFRangeDataset(STF_path, 
                                         train=False, 
                                         used_feature=used_feature, 
                                         width=width, 
                                         grid_sizes=grid_sizes, 
                                         pc_range=pc_range, 
                                         log=log,
                                         inverse=inverse,
                                         downsample=downsample,
                                         inpainting=inpainting,
                                         coord=coord,
                                        dataset_cfg=dataset_cfg)
        self.train_dataset = dataset
        self.test_dataset = test_dataset
