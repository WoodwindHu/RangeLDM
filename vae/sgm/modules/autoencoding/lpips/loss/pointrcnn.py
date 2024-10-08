
from easydict import EasyDict

import torch
import torch.nn as nn

from pcdet.ops.pointnet2.pointnet2_batch import pointnet2_modules
from pcdet.ops.pointnet2.pointnet2_stack import pointnet2_modules as pointnet2_modules_stack
from pcdet.ops.pointnet2.pointnet2_stack import pointnet2_utils as pointnet2_utils_stack
from ..util import get_ckpt_path


class PointNet2MSG(nn.Module):
    def __init__(self, model_cfg, input_channels, model_name, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg

        self.SA_modules = nn.ModuleList()
        channel_in = input_channels - 3

        self.num_points_each_layer = []
        skip_channel_list = [input_channels - 3]
        for k in range(self.model_cfg.SA_CONFIG.NPOINTS.__len__()):
            mlps = self.model_cfg.SA_CONFIG.MLPS[k].copy()
            channel_out = 0
            for idx in range(mlps.__len__()):
                mlps[idx] = [channel_in] + mlps[idx]
                channel_out += mlps[idx][-1]

            self.SA_modules.append(
                pointnet2_modules.PointnetSAModuleMSG(
                    npoint=self.model_cfg.SA_CONFIG.NPOINTS[k],
                    radii=self.model_cfg.SA_CONFIG.RADIUS[k],
                    nsamples=self.model_cfg.SA_CONFIG.NSAMPLE[k],
                    mlps=mlps,
                    use_xyz=self.model_cfg.SA_CONFIG.get('USE_XYZ', True),
                )
            )
            skip_channel_list.append(channel_out)
            channel_in = channel_out

        self.FP_modules = nn.ModuleList()

        for k in range(self.model_cfg.FP_MLPS.__len__()):
            pre_channel = self.model_cfg.FP_MLPS[k + 1][-1] if k + 1 < len(self.model_cfg.FP_MLPS) else channel_out
            self.FP_modules.append(
                pointnet2_modules.PointnetFPModule(
                    mlp=[pre_channel + skip_channel_list[k]] + self.model_cfg.FP_MLPS[k]
                )
            )

        self.num_point_features = self.model_cfg.FP_MLPS[0][-1]
        self.load_from_pretrained(name=model_name)
        for param in self.parameters():
            param.requires_grad = False

    def load_from_pretrained(self, name="pointnet2msg"):
        ckpt = get_ckpt_path(name, "sgm/modules/autoencoding/lpips/loss")
        self.load_state_dict(
            torch.load(ckpt, map_location=torch.device("cpu")), strict=False
        )
        print("loaded pretrained LPIPS loss from {}".format(ckpt))

    def forward(self, points):
        """
        Args:
            points: (B, num_points, 3 + C)
        Returns:
            batch_dict:
                l_features: list of (B, C, N)
        """
        batch_size, _, channels = points.shape
        xyz = points[:,:,:3].contiguous()
        features = points[:,:,3:].permute(0, 2, 1).contiguous() if channels > 3 else None

        l_xyz, l_features = [xyz], [features]
        for i in range(len(self.SA_modules)):
            li_xyz, li_features = self.SA_modules[i](l_xyz[i], l_features[i])
            l_xyz.append(li_xyz)
            l_features.append(li_features)

        for i in range(-1, -(len(self.FP_modules) + 1), -1):
            l_features[i - 1] = self.FP_modules[i](
                l_xyz[i - 1], l_xyz[i], l_features[i - 1], l_features[i]
            )  # (B, C, N)

        return l_features

class LPIPS_3d(nn.Module):
    def __init__(self, all_loss=False, channels=4, model_name='pointnet2msg'):
        super().__init__()
        cfg = {'SA_CONFIG':{
            'NPOINTS': [4096, 1024, 256, 64],
            'RADIUS': [[0.1, 0.5], [0.5, 1.0], [1.0, 2.0], [2.0, 4.0]],
            'NSAMPLE': [[16, 32], [16, 32], [16, 32], [16, 32]],
            'MLPS': [[[16, 16, 32], [32, 32, 64]],
                    [[64, 64, 128], [64, 96, 128]],
                    [[128, 196, 256], [128, 196, 256]],
                    [[256, 256, 512], [256, 384, 512]]]
            },
            'FP_MLPS': [[128, 128], [256, 256], [512, 512], [512, 512]]}
        cfg = EasyDict(cfg)
        self.pointnet2msg = PointNet2MSG(cfg, channels, model_name=model_name)
        self.all_loss = all_loss

    def forward(self, inputs, targets):
        inputs = self.pointnet2msg(inputs)
        targets = self.pointnet2msg(targets)
        if self.all_loss:
            losses = [torch.mean((inputs[i] - targets[i])**2, dim=[1,2], keepdim=True) for i in range(len(inputs))]
            loss = losses[0]
            for i in range(1, len(inputs)):
                loss += losses[i]
        else:
            losses = [torch.mean((inputs[i] - targets[i])**2, dim=1, keepdim=True) for i in range(len(inputs))]
            loss = losses[0]
        return loss