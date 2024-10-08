
from diffusers.pipelines.stable_diffusion.convert_from_ckpt import renew_vae_resnet_paths, assign_to_checkpoint, renew_vae_attention_paths, conv_attn_to_linear
from diffusers.utils import is_accelerate_available
from contextlib import nullcontext
from diffusers.models import AutoencoderKL
from omegaconf import OmegaConf
import torch
from utils import replace_conv, replace_attn, replace_down

if is_accelerate_available():
    from accelerate import init_empty_weights
    from accelerate.utils import set_module_tensor_to_device

def convert_ldm_vae_checkpoint(checkpoint, config):
    # extract state dict for VAE
    vae_state_dict = {}
    keys = list(checkpoint.keys())
    vae_key = "first_stage_model." if any(k.startswith("first_stage_model.") for k in keys) else ""
    for key in keys:
        if key.startswith(vae_key):
            vae_state_dict[key.replace(vae_key, "")] = checkpoint.get(key)

    new_checkpoint = {}

    new_checkpoint["encoder.conv_in.weight"] = vae_state_dict["encoder.conv_in.weight"]
    new_checkpoint["encoder.conv_in.bias"] = vae_state_dict["encoder.conv_in.bias"]
    new_checkpoint["encoder.conv_out.weight"] = vae_state_dict["encoder.conv_out.weight"]
    new_checkpoint["encoder.conv_out.bias"] = vae_state_dict["encoder.conv_out.bias"]
    new_checkpoint["encoder.conv_norm_out.weight"] = vae_state_dict["encoder.norm_out.weight"]
    new_checkpoint["encoder.conv_norm_out.bias"] = vae_state_dict["encoder.norm_out.bias"]

    new_checkpoint["decoder.conv_in.weight"] = vae_state_dict["decoder.conv_in.weight"]
    new_checkpoint["decoder.conv_in.bias"] = vae_state_dict["decoder.conv_in.bias"]
    new_checkpoint["decoder.conv_out.weight"] = vae_state_dict["decoder.conv_out.weight"]
    new_checkpoint["decoder.conv_out.bias"] = vae_state_dict["decoder.conv_out.bias"]
    new_checkpoint["decoder.conv_norm_out.weight"] = vae_state_dict["decoder.norm_out.weight"]
    new_checkpoint["decoder.conv_norm_out.bias"] = vae_state_dict["decoder.norm_out.bias"]

    try:
        new_checkpoint["quant_conv.weight"] = vae_state_dict["quant_conv.weight"]
        new_checkpoint["quant_conv.bias"] = vae_state_dict["quant_conv.bias"]
        new_checkpoint["post_quant_conv.weight"] = vae_state_dict["post_quant_conv.weight"]
        new_checkpoint["post_quant_conv.bias"] = vae_state_dict["post_quant_conv.bias"]
    except:
        print('no quant_conv')

    # Retrieves the keys for the encoder down blocks only
    num_down_blocks = len({".".join(layer.split(".")[:3]) for layer in vae_state_dict if "encoder.down" in layer})
    down_blocks = {
        layer_id: [key for key in vae_state_dict if f"down.{layer_id}" in key] for layer_id in range(num_down_blocks)
    }

    # Retrieves the keys for the decoder up blocks only
    num_up_blocks = len({".".join(layer.split(".")[:3]) for layer in vae_state_dict if "decoder.up" in layer})
    up_blocks = {
        layer_id: [key for key in vae_state_dict if f"up.{layer_id}" in key] for layer_id in range(num_up_blocks)
    }

    for i in range(num_down_blocks):
        resnets = [key for key in down_blocks[i] if f"down.{i}" in key and f"down.{i}.downsample" not in key]

        if f"encoder.down.{i}.downsample.conv.weight" in vae_state_dict:
            new_checkpoint[f"encoder.down_blocks.{i}.downsamplers.0.conv.weight"] = vae_state_dict.pop(
                f"encoder.down.{i}.downsample.conv.weight"
            )
            new_checkpoint[f"encoder.down_blocks.{i}.downsamplers.0.conv.bias"] = vae_state_dict.pop(
                f"encoder.down.{i}.downsample.conv.bias"
            )

        paths = renew_vae_resnet_paths(resnets)
        meta_path = {"old": f"down.{i}.block", "new": f"down_blocks.{i}.resnets"}
        assign_to_checkpoint(paths, new_checkpoint, vae_state_dict, additional_replacements=[meta_path], config=config)

    mid_resnets = [key for key in vae_state_dict if "encoder.mid.block" in key]
    num_mid_res_blocks = 2
    for i in range(1, num_mid_res_blocks + 1):
        resnets = [key for key in mid_resnets if f"encoder.mid.block_{i}" in key]

        paths = renew_vae_resnet_paths(resnets)
        meta_path = {"old": f"mid.block_{i}", "new": f"mid_block.resnets.{i - 1}"}
        assign_to_checkpoint(paths, new_checkpoint, vae_state_dict, additional_replacements=[meta_path], config=config)

    mid_attentions = [key for key in vae_state_dict if "encoder.mid.attn" in key]
    paths = renew_vae_attention_paths(mid_attentions)
    meta_path = {"old": "mid.attn_1", "new": "mid_block.attentions.0"}
    assign_to_checkpoint(paths, new_checkpoint, vae_state_dict, additional_replacements=[meta_path], config=config)
    conv_attn_to_linear(new_checkpoint)

    for i in range(num_up_blocks):
        block_id = num_up_blocks - 1 - i
        resnets = [
            key for key in up_blocks[block_id] if f"up.{block_id}" in key and f"up.{block_id}.upsample" not in key
        ]

        if f"decoder.up.{block_id}.upsample.conv.weight" in vae_state_dict:
            new_checkpoint[f"decoder.up_blocks.{i}.upsamplers.0.conv.weight"] = vae_state_dict[
                f"decoder.up.{block_id}.upsample.conv.weight"
            ]
            new_checkpoint[f"decoder.up_blocks.{i}.upsamplers.0.conv.bias"] = vae_state_dict[
                f"decoder.up.{block_id}.upsample.conv.bias"
            ]

        paths = renew_vae_resnet_paths(resnets)
        meta_path = {"old": f"up.{block_id}.block", "new": f"up_blocks.{i}.resnets"}
        assign_to_checkpoint(paths, new_checkpoint, vae_state_dict, additional_replacements=[meta_path], config=config)

    mid_resnets = [key for key in vae_state_dict if "decoder.mid.block" in key]
    num_mid_res_blocks = 2
    for i in range(1, num_mid_res_blocks + 1):
        resnets = [key for key in mid_resnets if f"decoder.mid.block_{i}" in key]

        paths = renew_vae_resnet_paths(resnets)
        meta_path = {"old": f"mid.block_{i}", "new": f"mid_block.resnets.{i - 1}"}
        assign_to_checkpoint(paths, new_checkpoint, vae_state_dict, additional_replacements=[meta_path], config=config)

    mid_attentions = [key for key in vae_state_dict if "decoder.mid.attn" in key]
    paths = renew_vae_attention_paths(mid_attentions)
    meta_path = {"old": "mid.attn_1", "new": "mid_block.attentions.0"}
    assign_to_checkpoint(paths, new_checkpoint, vae_state_dict, additional_replacements=[meta_path], config=config)
    conv_attn_to_linear(new_checkpoint)
    return new_checkpoint

def create_vae_diffusers_config(original_config, image_size: int):
    """
    Creates a config for the diffusers based on the config of the LDM model.
    """
    try: 
        vae_params = original_config.model.params.ddconfig
    except:
        vae_params = original_config.model.params.encoder_config.params


    block_out_channels = [vae_params.ch * mult for mult in vae_params.ch_mult]
    down_block_types = ["DownEncoderBlock2D"] * len(block_out_channels)
    up_block_types = ["UpDecoderBlock2D"] * len(block_out_channels)

    config = {
        "sample_size": image_size,
        "in_channels": vae_params.in_channels,
        "out_channels": vae_params.out_ch,
        "down_block_types": tuple(down_block_types),
        "up_block_types": tuple(up_block_types),
        "block_out_channels": tuple(block_out_channels),
        "latent_channels": vae_params.z_channels,
        "layers_per_block": vae_params.num_res_blocks,
    }
    return config

def convert_vae_to_diffusers(original_config, image_size, checkpoint, return_cfg=True):
    # Convert the VAE model.
    try:
        attn_free = original_config.model.params.ddconfig.attn_type == 'none'
    except:
        attn_free = original_config.model.params.encoder_config.params.attn_type == 'none'
        without_quant_conv = True
    vae_config = create_vae_diffusers_config(original_config, image_size=image_size)
    converted_vae_checkpoint = convert_ldm_vae_checkpoint(checkpoint, vae_config)

    if (
        "model" in original_config
        and "params" in original_config.model
        and "scale_factor" in original_config.model.params
    ):
        vae_scaling_factor = original_config.model.params.scale_factor
    else:
        vae_scaling_factor = 0.18215  # default SD scaling factor

    vae_config["scaling_factor"] = vae_scaling_factor

    ctx = init_empty_weights if is_accelerate_available() else nullcontext
    with ctx():
        vae = AutoencoderKL(**vae_config)
    if without_quant_conv:
        vae.quant_conv = torch.nn.Identity()
        vae.post_quant_conv = torch.nn.Identity()
    replace_down(vae)
    replace_conv(vae)
    if attn_free:
        replace_attn(vae)

    if is_accelerate_available():
        for param_name, param in converted_vae_checkpoint.items():
            set_module_tensor_to_device(vae, param_name, "cpu", value=param)
    else:
        vae.load_state_dict(converted_vae_checkpoint)
    if return_cfg:
        return vae, vae_config
    else:
        return vae
    


if __name__=='__main__':
    import numpy as np
    from PIL import Image
    import argparse
    from tqdm import tqdm
    parser = argparse.ArgumentParser(description="lidar point cloud generator.")
    parser.add_argument(
        "--cfg",
        type=str,
    )
    args = parser.parse_args()
    args = OmegaConf.load(args.cfg)
    args.out = 'test_convert_vae_vanilla'
    args.vae_config = 'vae_models/kitti-range-image-4xDown-attenfree-silu-circular-scale_lr-kl1e-6-bs16-big.yaml'
    args.vae_checkpoint = 'vae_models/2023-10-17T03-07-12_example_training-autoencoder-kl-f4-kitti-range-image-4xDown-attenfree-silu-circular-scale_lr-kl1e-6-bs16-big-epoch=000617.ckpt'
    checkpoint = torch.load(args.vae_checkpoint, map_location='cpu')['state_dict']

    config = OmegaConf.load(args.vae_config)
    vae, vae_config = convert_vae_to_diffusers(config, (1024, 64), checkpoint)
    from kitti360_range_image import KITTIRangeLoader
    import os
    os.makedirs(args.out, exist_ok=True)
    os.makedirs(f'{args.out}/input', exist_ok=True)
    os.makedirs(f'{args.out}/output', exist_ok=True)
    batch_size=4
    loader = KITTIRangeLoader(os.environ.get('KITTI360_DATASET'), 
                                        batch_size = batch_size, 
                                        num_workers=6, 
                                        )
    test_dataloader = loader.test_dataloader()
    to_range = loader.train_dataset.to_range_image
    test_dataloader_iter = iter(test_dataloader)
    psnr = 0
    error = 0
    vae = vae.cuda()
    vae.eval()
    for i in tqdm(range(1000//batch_size + 1)):
        batch = next(test_dataloader_iter)['jpg'].cuda()
        rec = vae(batch).sample
        pc_in_all= to_range.to_pc_torch(batch)
        bev_in_all = to_range.to_voxel(batch)
        pc_out_all= to_range.to_pc_torch(rec)
        bev_out_all = to_range.to_voxel(rec)
        for j in range(batch_size):
            if i*batch_size + j >= 1000:
                break
            # normalize to [0,1]
            input = batch[j]
            input[0] = (input[0] * to_range.std + to_range.mean) / to_range.range_fill_value[0]
            output = rec[j]
            output[0] = (output[0] * to_range.std + to_range.mean) / to_range.range_fill_value[0]
            error += torch.mean(torch.abs(input - output)).item()
            mse = torch.mean((input - output)**2).item()
            psnr += 10*np.log10(1./mse)
            pc = pc_out_all[j].cpu().detach().numpy()
            depth = np.linalg.norm(pc[:,:3], 2, axis=1)
            mask = depth < 70.0
            pc[mask, :].tofile(f'{args.out}/output/{i*batch_size + j}.bin')
            bev_image = Image.fromarray((bev_out_all[j].permute(2, 1, 0).cpu().detach().numpy().clip(0, 1)*255.).astype(np.uint8)[:,:,0], mode='L')
            bev_image.save(f'{args.out}/output/{i*batch_size + j}.png')
            pc = pc_in_all[j].cpu().detach().numpy()
            depth = np.linalg.norm(pc[:,:3], 2, axis=1)
            mask = depth < 70.0
            pc[mask, :].tofile(f'{args.out}/input/{i*batch_size + j}.bin')
            bev_image = Image.fromarray((bev_in_all[j].permute(2, 1, 0).cpu().detach().numpy().clip(0, 1)*255.).astype(np.uint8)[:,:,0], mode='L')
            bev_image.save(f'{args.out}/input/{i*batch_size + j}.png')
    print(f'MAE: {error/1000}')
    print(f'PSNR: {psnr/1000}')

    import pytorch3d.loss.chamfer  as chamfer
    cd = 0
    for i in tqdm(range(1000)):
        pts_in = np.fromfile(f'{args.out}/input/{i}.bin', dtype=np.float32).reshape(-1, 4)
        pts_out = np.fromfile(f'{args.out}/output/{i}.bin', dtype=np.float32).reshape(-1, 4)
        pts_in = torch.from_numpy(pts_in[:,:3]).cuda()
        pts_out = torch.from_numpy(pts_out[:,:3]).cuda()
        loss = chamfer.chamfer_distance(pts_in.unsqueeze(0), pts_out.unsqueeze(0))
        cd += loss[0].item()
    print(f'CD: {cd/1000}')