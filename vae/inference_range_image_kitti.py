import os
from PIL import Image

from omegaconf import OmegaConf
from sgm.util import exists, instantiate_from_config, isheatmap
from sgm.data.kitti360_range_image import point_cloud_to_range_image_KITTI
import numpy as np
import argparse
from pathlib import Path

parser = argparse.ArgumentParser(description='arg parser')
parser.add_argument('--cfg', type=str, default=None, help='config file')
parser.add_argument('--ckpt', type=str, default='2023-09-18T22-29-22_example_training-autoencoder-kl-f4-kitti-range-image-perceptual-dense-epoch=000078.ckpt', help='ckpt file')
args = parser.parse_args()

args.cfg = args.cfg if args.cfg else f'configs/example_training/autoencoder/kl-f4/{args.ckpt.split("/")[1][55:]}.yaml'


config = OmegaConf.load(args.cfg)
model = instantiate_from_config(config.model)
import torch
a = torch.load(args.ckpt, map_location=torch.device('cpu'))
model.load_state_dict(a['state_dict'])
model = model # .cuda()
model.eval()
data = instantiate_from_config(config.data)
stf = data.test_dataloader()
batch = next(iter(stf))['jpg'] # .cuda()
to_range = data.test_dataset.to_range_image
output_path = Path(args.ckpt).parent / 'vae_outputs'
os.makedirs(output_path, exist_ok=True)
def process(batch, i):
    print('######processing input######')
    range_image = batch.squeeze(0).permute(2, 1, 0).cpu().detach().numpy() * 40 + 20
    save_range_image(range_image, f'{output_path}/range_gray{i}.png')
    pc= to_range.to_pc_torch(batch)
    pc[0, :,:4].cpu().detach().numpy().tofile(f'{output_path}/vae_input{i}.bin')
    bev = to_range.to_voxel(batch)
    bev_image = Image.fromarray((bev[0].permute(2, 1, 0).cpu().detach().numpy().clip(0, 1)*255.).astype(np.uint8)[:,:,0], mode='L')
    bev_image.save(f'{output_path}/bev{i}.png')
    print('######processing output######')
    z, dec, reg_log = model(batch)
    range_image = dec.squeeze(0).permute(2, 1, 0).cpu().detach().numpy() * 40 + 20
    save_range_image(range_image, f'{output_path}/range_gray_out{i}.png')
    pc= to_range.to_pc_torch(dec)
    pc[0, :,:4].cpu().detach().numpy().tofile(f'{output_path}/vae_output{i}.bin')
    bev = to_range.to_voxel(dec)
    bev_image = Image.fromarray((bev[0].permute(2, 1, 0).cpu().detach().numpy().clip(0, 1)*255.).astype(np.uint8)[:,:,0], mode='L')
    bev_image.save(f'{output_path}/bev_out{i}.png')

def save_range_image(range_image, fname):
    range_gray = Image.fromarray(((range_image[:,:,0].clip(0, 40))/40.*255.).astype(np.uint8), mode='L')
    range_gray.save(fname)

for i in range(batch.shape[0]):
    process(batch[i:i+1], i)

