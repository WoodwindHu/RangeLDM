import os
from convert_vae import convert_vae_to_diffusers
from diffusers import DDPMScheduler, UNet2DModel
from kitti360_range_image_vanilla import KITTIRangeVanillaLoader
from kitti360_range_image import KITTIRangeLoader
from nuscenes_range_image import nuScenesRangeLoader
from Waymo_range_image import WaymoRangeLoader
import torch
import numpy as np
# import open3d as o3d
import safetensors
import argparse
from tqdm import tqdm 
from PIL import Image
from pipelines import DDPMPipelineRange, DDIMPipelineRange, LDMPipelineRange
from omegaconf import OmegaConf
from diffusers.models import AutoencoderKL
from utils import replace_attn, replace_conv, replace_down
from accelerate import PartialState
from STF_range_image import STFRangeLoader

def parse_args():
    parser = argparse.ArgumentParser(description="lidar point cloud generator.")
    parser.add_argument("--cfg", type=str, required=True, help="The config of training process")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=1000,
    )
    args = parser.parse_args()
    samples = args.samples
    batch_size = args.batch_size
    cfg = args.cfg
    args = OmegaConf.load(cfg)
    if args.output_dir is None:
        cfg_name = cfg.split('/')[-1][:-5]
        args.output_dir = f'outputs/{cfg_name}'
    if batch_size:
        args.eval_batch_size = batch_size
    args.samples = samples
    args.unet_config = args.output_dir+'/unet/config.json'
    args.unet_checkpoint = args.output_dir+'/unet/diffusion_pytorch_model.safetensors'
    args.scheduler_config = args.output_dir+'/scheduler/scheduler_config.json'
    args.out = args.output_dir+'/generated'
    if args.with_vae:
        args.vae_config = args.output_dir+'/vae/config.json'
        args.vae_checkpoint = args.output_dir+'/vae/diffusion_pytorch_model.safetensors'
    return args

args = parse_args()
distributed_state = PartialState()

batch_size = args.eval_batch_size
if hasattr(args, 'nuscenes') and args.nuscenes:
    loader = nuScenesRangeLoader(os.environ.get('NUSCENES_DATASET'), 
                                        batch_size=batch_size, 
                                        num_workers=6, 
                                        )
elif hasattr(args, 'range_vanilla') and args.range_vanilla:
    loader = KITTIRangeVanillaLoader(os.environ.get('KITTI360_DATASET'), 
                                        batch_size=batch_size, 
                                        num_workers=6, 
                                        )
elif hasattr(args, 'STF'):
    loader = STFRangeLoader(**args.STF)
elif hasattr(args, 'Waymo'):
    loader = WaymoRangeLoader(**args.Waymo)
else:
    loader = KITTIRangeLoader(os.environ.get('KITTI360_DATASET'), 
                                        batch_size=batch_size, 
                                        num_workers=6, 
                                        )
if hasattr(args, 'range_mean'):
    loader.train_dataset.to_range_image.mean = args.range_mean
    loader.train_dataset.to_range_image.std = args.range_std
to_range = loader.train_dataset.to_range_image
# save_generated(np.load('/data/hqj/KITTI-360/range_images/2013_05_28_drive_0005_sync/velodyne_points/data/0000000000.npy'))

config = UNet2DModel.load_config(args.unet_config)
model = UNet2DModel.from_config(config)
if args.with_vae:
    config = AutoencoderKL.load_config(args.vae_config)
    vae = AutoencoderKL.from_config(config)
    vae_checkpoint = safetensors.torch.load_file(args.vae_checkpoint)
    if 'quant_conv.weight' not in vae_checkpoint:
        vae.quant_conv = torch.nn.Identity()
        vae.post_quant_conv = torch.nn.Identity()
    replace_down(vae)
    replace_conv(vae)
    if 'encoder.mid_block.attentions.0.to_q.weight' not in vae_checkpoint:
        replace_attn(vae)
    vae.load_state_dict(vae_checkpoint)
    total_params = sum(p.numel() for p in vae.parameters())
    if distributed_state.is_main_process:
        print(f"Parameters of VAE: {total_params/1024./1024.} M")

if hasattr(args, 'all_circonv') and args.all_circonv:
    replace_down(model)
    replace_conv(model)
elif hasattr(args, 'sub_circonv') and args.sub_circonv:
    for submodel in [model.down_blocks[0], 
                    model.down_blocks[1],
                    model.down_blocks[2],]:
        replace_down(submodel)
    for submodel in [model.conv_in, 
                    model.down_blocks[0], 
                    model.down_blocks[1],
                    model.down_blocks[2], 
                    model.up_blocks[3],
                    model.up_blocks[4], 
                    model.up_blocks[5],
                    model.conv_out]:
        replace_conv(submodel)
# breakpoint()
safetensors.torch.load_model(model, args.unet_checkpoint)

total_params = sum(p.numel() for p in model.parameters())
if distributed_state.is_main_process:
    print(f"Parameters of UNet: {total_params/1024./1024.} M")

config = DDPMScheduler.load_config(args.scheduler_config)
scheduler = DDPMScheduler.from_config(config)

if args.with_vae:
    Pipeline = LDMPipelineRange
    pipe = Pipeline(
                    vae=vae,
                    unet=model,
                    scheduler=scheduler,
                    pos_encoding=hasattr(args, 'pos_encoding') and args.pos_encoding,
                )
    generator = None
    # generator = torch.Generator(device=pipe.device) # .manual_seed(0)
    pipe = pipe.to(distributed_state.device)
else:
    if args.ddim:
        Pipeline = DDIMPipelineRange
    else:
        Pipeline = DDPMPipelineRange
    pipe = Pipeline(
                    unet=model,
                    scheduler=scheduler,
                    pos_encoding=hasattr(args, 'pos_encoding') and args.pos_encoding,
                )
    pipe = pipe.to(distributed_state.device)
    generator = None
    # generator = torch.Generator(device=pipe.device) # .manual_seed(0)

if not distributed_state.is_main_process:
    pipe.set_progress_bar_config(disable=True)

os.makedirs(args.out, exist_ok=True)

for i in tqdm(range(args.samples//batch_size//distributed_state.num_processes + 1), disable=not distributed_state.is_main_process):
    image = pipe(generator=generator,
                batch_size=batch_size,
                num_inference_steps=args.ddpm_num_inference_steps,
                output_type="torch",
                )
    if hasattr(args, 'scaling_factor'):
        image = image / args.scaling_factor
    if hasattr(args, 'shifting_factor'):
        image = image + args.shifting_factor
    if hasattr(args, 'Waymo'):
        image = image[:,:,3:2653,:]
    pc_all= to_range.to_pc_torch(image)
    bev_out_all = to_range.to_voxel(image)
    for j in range(batch_size):
        if (distributed_state.process_index+distributed_state.num_processes*i)*batch_size+j >= args.samples:
            break
        pc = pc_all[j].cpu().detach().numpy()
        depth = np.linalg.norm(pc[:,:3], 2, axis=1)
        mask = depth < 90.0
        pc[mask, :].tofile(f'{args.out}/{(distributed_state.process_index+distributed_state.num_processes*i)*batch_size+j}.bin')
        bev_image = Image.fromarray((bev_out_all[j].permute(2, 1, 0).cpu().detach().numpy().clip(0, 1)*255.).astype(np.uint8)[:,:,0], mode='L')
        bev_image.save(f'{args.out}/{(distributed_state.process_index+distributed_state.num_processes*i)*batch_size+j}.png')
        range_image = Image.fromarray((image[j].permute(2, 1, 0).cpu().detach().numpy().clip(0, 1)*255.).astype(np.uint8)[:,:,0], mode='L')
        range_image.save(f'{args.out}/{(distributed_state.process_index+distributed_state.num_processes*i)*batch_size+j}_range.png')
