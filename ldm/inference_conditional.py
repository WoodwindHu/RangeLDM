import os
from convert_vae import convert_vae_to_diffusers
from diffusers import DDPMScheduler, UNet2DModel
from encoders import SparseRangeImageEncoder2
from kitti360_range_image import KITTIRangeLoader
from nuscenes_range_image import nuScenesRangeLoader
import torch
import numpy as np
# import open3d as o3d
import safetensors
import argparse
from tqdm import tqdm 
from PIL import Image
from pipelines import DDPMPipelineRange, DDIMPipelineRange, LDMPipelineRange, LDMUpscalePipelineRange
from omegaconf import OmegaConf
from diffusers.models import AutoencoderKL
from utils import replace_attn, replace_conv, replace_down
from accelerate import PartialState, Accelerator

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
    args.vae_config = args.output_dir+'/vae/config.json'
    args.vae_checkpoint = args.output_dir+'/vae/diffusion_pytorch_model.safetensors'
    return args

args = parse_args()
distributed_state = PartialState()
accelerator = Accelerator()

batch_size = args.eval_batch_size
if hasattr(args, 'nuscenes') and args.nuscenes:
    loader = nuScenesRangeLoader(os.environ.get('NUSCENES_DATASET'),
                                batch_size=batch_size,
                                num_workers=6,
                                downsample=args.upsample,
                                inpainting=args.inpainting,
                                )
    range_limit = 90.0
else:
    loader = KITTIRangeLoader(os.environ.get('KITTI360_DATASET'), 
                                batch_size=batch_size, 
                                num_workers=6, 
                                downsample=args.upsample,
                                inpainting=args.inpainting
                                )
    range_limit = 70.0
if hasattr(args, 'range_mean'):
    loader.test_dataset.to_range_image.mean = args.range_mean
    loader.test_dataset.to_range_image.std = args.range_std
test_dataloader = loader.test_dataloader()
# test_dataloader = accelerator.prepare(test_dataloader)
test_dataset = loader.test_dataset
to_range = loader.test_dataset.to_range_image

config = UNet2DModel.load_config(args.unet_config)
model = UNet2DModel.from_config(config)
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

if args.upsample:
    condition_encoder = SparseRangeImageEncoder2()
    condition_encoder.to("cuda")
else:
    condition_encoder = None
pipe = LDMUpscalePipelineRange(
                unet=model,
                scheduler=scheduler,
                vae=vae,
            )
pipe = pipe.to(distributed_state.device)

if not distributed_state.is_main_process:
    pipe.set_progress_bar_config(disable=True)


key = 'down' if args.upsample else 'masked_image'
os.makedirs(args.out, exist_ok=True)
if args.upsample:
    result_path = f'{args.out}/densification_result'
    target_path = f'{args.out}/densification_target'
    input_path = f'{args.out}/densification_input'
elif args.inpainting:
    result_path = f'{args.out}/inpainting_result'
    target_path = f'{args.out}/inpainting_target'
    input_path = f'{args.out}/inpainting_input'
else:
    raise NotImplementedError
os.makedirs(result_path, exist_ok=True)
os.makedirs(target_path, exist_ok=True)
os.makedirs(input_path, exist_ok=True)

test_dataloader = iter(test_dataloader)
batch = next(test_dataloader)
batch = next(test_dataloader)
for i in tqdm(range(args.samples//batch_size//distributed_state.num_processes + 1), disable=not distributed_state.is_main_process):
    generator = torch.Generator().manual_seed(distributed_state.process_index+distributed_state.num_processes*i)
    images = pipe(
                image=batch[key],
                mask=None if args.upsample else batch['inpainting_mask'],
                condition_encoder=condition_encoder,
                generator=generator,
                batch_size=batch_size,
                num_inference_steps=args.ddpm_num_inference_steps,
                output_type="torch",
            )
    pc_gt_all = to_range.to_pc_torch(batch['jpg'])
    bev_gt_all = to_range.to_voxel(batch['jpg'])
    if args.inpainting:
        pc_in_all = to_range.to_pc_torch(batch['masked_image'])
        bev_in_all = to_range.to_voxel(batch['masked_image'])
    elif args.upsample:
        down_imag = -torch.ones_like(batch['jpg'])
        if isinstance(test_dataset.downsample, int):
            test_dataset.downsample = [1, test_dataset.downsample]
        down_imag[:, :, 
                    (test_dataset.downsample[0]//2)::test_dataset.downsample[0], 
                    (test_dataset.downsample[1]//2)::test_dataset.downsample[1]] = batch['down']
        pc_in_all = to_range.to_pc_torch(down_imag)
        bev_in_all = to_range.to_voxel(down_imag)
    else:
        raise NotImplementedError
    pc_all= to_range.to_pc_torch(images)
    bev_out_all = to_range.to_voxel(images)
    for j in range(batch_size):
        # if (distributed_state.process_index+distributed_state.num_processes*i)*batch_size+j >= args.samples:
        #     break
        pc = pc_all[j].cpu().detach().numpy()
        depth = np.linalg.norm(pc[:,:3], 2, axis=1)
        mask = depth < range_limit
        pc[mask, :].tofile(f'{result_path}/{j}_seed_{distributed_state.process_index+distributed_state.num_processes*i}.bin')
        bev_image = Image.fromarray((bev_out_all[j].permute(2, 1, 0).cpu().detach().numpy().clip(0, 1)*255.).astype(np.uint8)[:,:,0], mode='L')
        bev_image.save(f'{result_path}/{j}_seed_{distributed_state.process_index+distributed_state.num_processes*i}.png')
        if distributed_state.process_index+distributed_state.num_processes*i ==0:
            pc = pc_gt_all[j].cpu().detach().numpy()
            depth = np.linalg.norm(pc[:,:3], 2, axis=1)
            mask = depth < range_limit
            pc[mask, :].tofile(f'{target_path}/{j}_seed_{distributed_state.process_index+distributed_state.num_processes*i}.bin')
            bev_image = Image.fromarray((bev_gt_all[j].permute(2, 1, 0).cpu().detach().numpy().clip(0, 1)*255.).astype(np.uint8)[:,:,0], mode='L')
            bev_image.save(f'{target_path}/{j}_seed_{distributed_state.process_index+distributed_state.num_processes*i}.png')
            pc = pc_in_all[j].cpu().detach().numpy()
            depth = np.linalg.norm(pc[:,:3], 2, axis=1)
            mask = depth < range_limit
            pc[mask, :].tofile(f'{input_path}/{j}_seed_{distributed_state.process_index+distributed_state.num_processes*i}.bin')
            bev_image = Image.fromarray((bev_in_all[j].permute(2, 1, 0).cpu().detach().numpy().clip(0, 1)*255.).astype(np.uint8)[:,:,0], mode='L')
            bev_image.save(f'{input_path}/{j}_seed_{distributed_state.process_index+distributed_state.num_processes*i}.png')
