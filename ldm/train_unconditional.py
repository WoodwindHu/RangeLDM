import argparse
import inspect
import logging
import math
import os
import shutil
from datetime import timedelta
from pathlib import Path
from typing import Optional

import accelerate
from convert_vae import convert_vae_to_diffusers
import datasets
from nuscenes_range_image import nuScenesRangeLoader
from kitti360_range_image import KITTIRangeLoader
from STF_range_image import STFRangeLoader
from kitti360_range_image_vanilla import KITTIRangeVanillaLoader
import torch
import torch.nn.functional as F
from accelerate import Accelerator, InitProcessGroupKwargs
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration
from datasets import load_dataset, Dataset
from huggingface_hub import HfFolder, Repository, create_repo, whoami
from packaging import version
from torchvision import transforms
from tqdm.auto import tqdm

import diffusers
from diffusers import DDPMPipeline, DDPMScheduler, UNet2DModel
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from diffusers.utils import check_min_version, is_accelerate_version, is_tensorboard_available, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available

from glob import glob
import numpy as np
from pipelines import DDPMPipelineRange, DDIMPipelineRange, LDMPipelineRange
from transformers.utils import ContextManagers
from accelerate.state import AcceleratorState
from omegaconf import OmegaConf
import omegaconf

from PIL import Image

from utils import replace_down, replace_conv

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.21.0.dev0")

logger = get_logger(__name__, log_level="INFO")

def compute_snr(noise_scheduler, timesteps):
    """
    Computes SNR as per https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L847-L849
    """
    alphas_cumprod = noise_scheduler.alphas_cumprod
    sqrt_alphas_cumprod = alphas_cumprod**0.5
    sqrt_one_minus_alphas_cumprod = (1.0 - alphas_cumprod) ** 0.5

    # Expand the tensors.
    # Adapted from https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L1026
    sqrt_alphas_cumprod = sqrt_alphas_cumprod.to(device=timesteps.device)[timesteps].float()
    while len(sqrt_alphas_cumprod.shape) < len(timesteps.shape):
        sqrt_alphas_cumprod = sqrt_alphas_cumprod[..., None]
    alpha = sqrt_alphas_cumprod.expand(timesteps.shape)

    sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod.to(device=timesteps.device)[timesteps].float()
    while len(sqrt_one_minus_alphas_cumprod.shape) < len(timesteps.shape):
        sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod[..., None]
    sigma = sqrt_one_minus_alphas_cumprod.expand(timesteps.shape)

    # Compute SNR.
    snr = (alpha / sigma) ** 2
    return snr


def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.

    :param arr: the 1-D numpy array.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    if not isinstance(arr, torch.Tensor):
        arr = torch.from_numpy(arr)
    res = arr[timesteps].float().to(timesteps.device)
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument("--cfg", type=str, required=True, help="The config of training process")
    args = parser.parse_args()
    cfg = args.cfg
    args = OmegaConf.load(cfg)
    if args.output_dir is None:
        cfg_name = cfg.split('/')[-1][:-5]
        args.output_dir = f'outputs/{cfg_name}'
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank


    return args


def get_full_repo_name(model_id: str, organization: Optional[str] = None, token: Optional[str] = None):
    if token is None:
        token = HfFolder.get_token()
    if organization is None:
        username = whoami(token)["name"]
        return f"{username}/{model_id}"
    else:
        return f"{organization}/{model_id}"


def main(args):
    logging_dir = os.path.join(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=7200))  # a big number for high resolution or big dataset
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.logger,
        project_config=accelerator_project_config,
        kwargs_handlers=[kwargs],
    )

    if args.logger == "tensorboard":
        if not is_tensorboard_available():
            raise ImportError("Make sure to install tensorboard if you want to use it for logging during training.")

    elif args.logger == "wandb":
        if not is_wandb_available():
            raise ImportError("Make sure to install wandb if you want to use it for logging during training.")
        import wandb

    # `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):
        # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
        def save_model_hook(models, weights, output_dir):
            if args.use_ema:
                ema_model.save_pretrained(os.path.join(output_dir, "unet_ema"))

            for i, model in enumerate(models):
                model.save_pretrained(os.path.join(output_dir, "unet"))

                # make sure to pop weight so that corresponding model is not saved again
                weights.pop()

        def load_model_hook(models, input_dir):
            if args.use_ema:
                load_model = EMAModel.from_pretrained(os.path.join(input_dir, "unet_ema"), UNet2DModel)
                ema_model.load_state_dict(load_model.state_dict())
                ema_model.to(accelerator.device)
                del load_model

            for i in range(len(models)):
                # pop models so that they are not loaded again
                model = models.pop()

                # load diffusers style into model
                load_model = UNet2DModel.from_pretrained(input_dir, subfolder="unet")
                model.register_to_config(**load_model.config)

                model.load_state_dict(load_model.state_dict())
                del load_model

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.push_to_hub:
            if args.hub_model_id is None:
                repo_name = get_full_repo_name(Path(args.output_dir).name, token=args.hub_token)
            else:
                repo_name = args.hub_model_id
            create_repo(repo_name, exist_ok=True, token=args.hub_token)
            repo = Repository(args.output_dir, clone_from=repo_name, token=args.hub_token)

            with open(os.path.join(args.output_dir, ".gitignore"), "w+") as gitignore:
                if "step_*" not in gitignore:
                    gitignore.write("step_*\n")
                if "epoch_*" not in gitignore:
                    gitignore.write("epoch_*\n")
        elif args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    if args.with_vae:
        def deepspeed_zero_init_disabled_context_manager():
            """
            returns either a context list that includes one that will disable zero.Init or an empty context list
            """
            deepspeed_plugin = AcceleratorState().deepspeed_plugin if accelerate.state.is_initialized() else None
            if deepspeed_plugin is None:
                return []

            return [deepspeed_plugin.zero3_init_context_manager(enable=False)]

        # Currently Accelerate doesn't know how to handle multiple models under Deepspeed ZeRO stage 3.
        # For this to work properly all models must be run through `accelerate.prepare`. But accelerate
        # will try to assign the same optimizer with the same weights to all models during
        # `deepspeed.initialize`, which of course doesn't work.
        #
        # For now the following workaround will partially support Deepspeed ZeRO-3, by excluding the 2
        # frozen models from being partitioned during `zero.Init` which gets called during
        # `from_pretrained` So CLIPTextModel and AutoencoderKL will not enjoy the parameter sharding
        # across multiple gpus and only UNet2DConditionModel will get ZeRO sharded.
        vae_config = OmegaConf.load(args.vae_config)
        vae_checkpoint = torch.load(args.vae_checkpoint, map_location='cpu')['state_dict']
        with ContextManagers(deepspeed_zero_init_disabled_context_manager()):
            vae, vae_config = convert_vae_to_diffusers(vae_config, tuple(args.resolution), vae_checkpoint, return_cfg=True)

    # Initialize the model
    if args.model_config:
        model_config = dict(args.model_config)
        for key in model_config:
            if type(model_config[key]) == omegaconf.listconfig.ListConfig:
                model_config[key] = tuple(model_config[key])
        model = UNet2DModel(**model_config)
    elif args.model_config_name_or_path:
        config = UNet2DModel.load_config(args.model_config_name_or_path)
        model = UNet2DModel.from_config(config)
    elif not args.with_vae:
        model = UNet2DModel(
            sample_size=tuple(args.resolution),
            in_channels=2,
            out_channels=2,
            layers_per_block=2,
            block_out_channels=tuple(args.block_out_channels) if args.block_out_channels else (64, 64, 128, 128, 256, 256),
            down_block_types=(
                "DownBlock2D",
                "DownBlock2D",
                "DownBlock2D",
                "DownBlock2D",
                "AttnDownBlock2D",
                "DownBlock2D",
            ),
            up_block_types=(
                "UpBlock2D",
                "AttnUpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
            ),
        ) 
    else: 
        model = UNet2DModel(
            sample_size=(args.resolution[0]//4, args.resolution[1]//4), # 4x downsample
            in_channels=vae_config['latent_channels'],
            out_channels=vae_config['latent_channels'],
            layers_per_block=2,
            block_out_channels=tuple(args.block_out_channels) if args.block_out_channels else (128, 128, 256, 256),
            down_block_types=(
                "DownBlock2D",
                "DownBlock2D",
                "AttnDownBlock2D",
                "DownBlock2D",
            ),
            up_block_types=(
                "UpBlock2D",
                "AttnUpBlock2D",
                "UpBlock2D",
                "UpBlock2D",
            ),
        )
        

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

    if accelerator.is_main_process:
        print(model)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Number of unet parameters: {total_params/1024./1024.} M")
        if args.with_vae:
            total_params_vae = sum(p.numel() for p in vae.parameters())
            print(f"Number of vae parameters: {total_params_vae/1024./1024.} M")
            print(f"Number of total parameters: {(total_params+total_params_vae)/1024./1024.} M")

    # Create EMA for the model.
    if args.use_ema:
        ema_model = EMAModel(
            model.parameters(),
            decay=args.ema_max_decay,
            use_ema_warmup=True,
            inv_gamma=args.ema_inv_gamma,
            power=args.ema_power,
            model_cls=UNet2DModel,
            model_config=model.config,
        )

    if args.enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warn(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            model.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    # Initialize the scheduler
    accepts_prediction_type = "prediction_type" in set(inspect.signature(DDPMScheduler.__init__).parameters.keys())
    if accepts_prediction_type:
        noise_scheduler = DDPMScheduler(
            num_train_timesteps=args.ddpm_num_steps,
            beta_schedule=args.ddpm_beta_schedule,
            prediction_type=args.prediction_type,
            clip_sample=False,
        )
    else:
        noise_scheduler = DDPMScheduler(num_train_timesteps=args.ddpm_num_steps, beta_schedule=args.ddpm_beta_schedule, clip_sample=False, )

    # Initialize the optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # Get the datasets: you can either provide your own training and evaluation files (see below)
    # or specify a Dataset from the hub (the dataset will be downloaded automatically from the datasets Hub).



    if hasattr(args, 'nuscenes') and args.nuscenes:
        loader = nuScenesRangeLoader(os.environ.get('NUSCENES_DATASET'), 
                                        batch_size=args.train_batch_size, 
                                        num_workers=args.dataloader_num_workers, 
                                        )
    elif hasattr(args, 'range_vanilla') and args.range_vanilla:
        loader = KITTIRangeVanillaLoader(os.environ.get('KITTI360_DATASET'), 
                                        batch_size=args.train_batch_size, 
                                        num_workers=args.dataloader_num_workers, 
                                        )
    elif hasattr(args, 'STF'):
        loader = STFRangeLoader(**args.STF)
    else:
        loader = KITTIRangeLoader(os.environ.get('KITTI360_DATASET'), 
                                        batch_size=args.train_batch_size, 
                                        num_workers=args.dataloader_num_workers, 
                                        )
    if hasattr(args, 'range_mean'):
        loader.train_dataset.to_range_image.mean = args.range_mean
        loader.train_dataset.to_range_image.std = args.range_std
    train_dataloader = loader.train_dataloader()
    to_range = loader.train_dataset.to_range_image

    # Initialize the learning rate scheduler
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=(len(train_dataloader) * args.num_epochs),
    )

    # Prepare everything with our `accelerator`.
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )
    if args.with_vae:
        vae.to(accelerator.device)
    if args.use_ema:
        ema_model.to(accelerator.device)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        run = os.path.split(__file__)[-1].split(".")[0]
        accelerator.init_trackers(run)

    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    max_train_steps = args.num_epochs * num_update_steps_per_epoch

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(loader.train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {max_train_steps}")

    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            resume_global_step = global_step * args.gradient_accumulation_steps
            first_epoch = global_step // num_update_steps_per_epoch
            resume_step = resume_global_step % (num_update_steps_per_epoch * args.gradient_accumulation_steps)
    if hasattr(args, 'pos_encoding') and args.pos_encoding:
        if args.with_vae:
            shape = [args.train_batch_size, 1, args.resolution[0]//4, args.resolution[1]//4]
        else:
            shape = [args.train_batch_size, 1, args.resolution[0], args.resolution[1]]
        pos_encoding = torch.zeros(
            shape, dtype=(torch.float32 if args.mixed_precision == "no" else torch.float16)
        ).to(accelerator.device)
        pos_encoding[:,:,0,:] = 1

    # Train!
    for epoch in range(first_epoch, args.num_epochs):
        model.train()
        progress_bar = tqdm(total=num_update_steps_per_epoch, disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")
        for step, batch in enumerate(train_dataloader):
            # Skip steps until we reach the resumed step
            if args.resume_from_checkpoint and epoch == first_epoch and step < resume_step:
                if step % args.gradient_accumulation_steps == 0:
                    progress_bar.update(1)
                continue
            
            clean_images = batch["jpg"]
            if args.with_vae:
                # Convert images to latent space
                clean_images = vae.encode(clean_images.to(torch.float32 if args.mixed_precision == "no" else torch.float16)).latent_dist.sample()
                clean_images = clean_images * vae.config.scaling_factor
            elif hasattr(args, 'scaling_factor'):
                if hasattr(args, 'shifting_factor'):
                    clean_images = clean_images - args.shifting_factor
                clean_images = clean_images * args.scaling_factor
            # Sample noise that we'll add to the images
            noise = torch.randn(
                clean_images.shape, dtype=(torch.float32 if args.mixed_precision == "no" else torch.float16)
            ).to(clean_images.device)
            bsz = clean_images.shape[0]
            # Sample a random timestep for each image
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (bsz,), device=clean_images.device
            ).long()

            # Add noise to the clean images according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)

            if hasattr(args, 'pos_encoding') and args.pos_encoding:
                noisy_images = torch.cat([noisy_images, pos_encoding[:noisy_images.shape[0]]], dim=1)

            with accelerator.accumulate(model):
                # Predict the noise residual
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(clean_images, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")
                
                model_output = model(noisy_images, timesteps).sample

                # if args.prediction_type == "epsilon":
                #     loss = F.mse_loss(model_output, noise)  # this could have different weights!
                # elif args.prediction_type == "sample":
                #     alpha_t = _extract_into_tensor(
                #         noise_scheduler.alphas_cumprod, timesteps, (clean_images.shape[0], 1, 1, 1)
                #     )
                #     snr_weights = alpha_t / (1 - alpha_t)
                #     loss = snr_weights * F.mse_loss(
                #         model_output, clean_images, reduction="none"
                #     )  # use SNR weighting from distillation paper
                #     loss = loss.mean()
                # else:
                #     raise ValueError(f"Unsupported prediction type: {args.prediction_type}")
                if args.snr_gamma is None:
                    loss = F.mse_loss(model_output.float(), target.float(), reduction="mean")
                else:
                    # Compute loss-weights as per Section 3.4 of https://arxiv.org/abs/2303.09556.
                    # Since we predict the noise instead of x_0, the original formulation is slightly changed.
                    # This is discussed in Section 4.2 of the same paper.
                    snr = compute_snr(noise_scheduler, timesteps)
                    if noise_scheduler.config.prediction_type == "v_prediction":
                        # Velocity objective requires that we add one to SNR values before we divide by them.
                        snr = snr + 1
                    mse_loss_weights = (
                        torch.stack([snr, args.snr_gamma * torch.ones_like(timesteps)], dim=1).min(dim=1)[0] / snr
                    )

                    loss = F.mse_loss(model_output.float(), target.float(), reduction="none")
                    loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
                    loss = loss.mean()

                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                if args.use_ema:
                    ema_model.step(model.parameters())
                progress_bar.update(1)
                global_step += 1

                if global_step % args.checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        if args.checkpoints_total_limit is not None:
                            checkpoints = os.listdir(args.output_dir)
                            checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
                            checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

                            # before we save the new checkpoint, we need to have at _most_ `checkpoints_total_limit - 1` checkpoints
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                removing_checkpoints = checkpoints[0:num_to_remove]

                                logger.info(
                                    f"{len(checkpoints)} checkpoints already exist, removing {len(removing_checkpoints)} checkpoints"
                                )
                                logger.info(f"removing checkpoints: {', '.join(removing_checkpoints)}")

                                for removing_checkpoint in removing_checkpoints:
                                    removing_checkpoint = os.path.join(args.output_dir, removing_checkpoint)
                                    shutil.rmtree(removing_checkpoint)

                    
                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "step": global_step}
            if args.use_ema:
                logs["ema_decay"] = ema_model.cur_decay_value
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
        progress_bar.close()

        accelerator.wait_for_everyone()

        # Generate sample images for visual inspection
        if accelerator.is_main_process:
            if epoch % args.save_images_epochs == 0 or epoch == args.num_epochs - 1:
                unet = accelerator.unwrap_model(model)

                if args.use_ema:
                    ema_model.store(unet.parameters())
                    ema_model.copy_to(unet.parameters())

                if args.with_vae:
                    pipeline = LDMPipelineRange(
                        unet=unet,
                        scheduler=noise_scheduler,
                        vae=vae,
                        pos_encoding=hasattr(args, 'pos_encoding') and args.pos_encoding,
                    )
                    generator = torch.Generator().manual_seed(0)
                else:
                    pipeline = DDPMPipelineRange(
                        unet=unet,
                        scheduler=noise_scheduler,
                    ) if not args.ddim else DDIMPipelineRange(
                        unet=unet,
                        scheduler=noise_scheduler,
                        pos_encoding=hasattr(args, 'pos_encoding') and args.pos_encoding,
                    )
                    generator = torch.Generator(device=pipeline.device).manual_seed(0)
                # run pipeline in inference (sample random noise and denoise)
                images = pipeline(
                    generator=generator,
                    batch_size=args.eval_batch_size,
                    num_inference_steps=args.ddpm_num_inference_steps,
                    output_type="torch",
                )

                if hasattr(args, 'scaling_factor'):
                    images = images / args.scaling_factor
                if hasattr(args, 'shifting_factor'):
                    images = images + args.shifting_factor


                if args.use_ema:
                    ema_model.restore(unet.parameters())

                # from lidar_utils import save_generated
                out_path = os.path.join(args.output_dir, f"epoch-{epoch}")
                os.makedirs(out_path, exist_ok=True)
                pc_all= to_range.to_pc_torch(images)
                bev_out_all = to_range.to_voxel(images)
                for j in range(args.eval_batch_size):
                    pc = pc_all[j].cpu().detach().numpy()
                    depth = np.linalg.norm(pc[:,:3], 2, axis=1)
                    mask = depth < 80.0
                    pc[mask, :].tofile(f'{out_path}/{j}.bin')
                    bev_image = Image.fromarray((bev_out_all[j].permute(2, 1, 0).cpu().detach().numpy().clip(0, 1)*255.).astype(np.uint8)[:,:,0], mode='L')
                    bev_image.save(f'{out_path}/bev_out{j}.png')
                del pipeline

            if epoch % args.save_model_epochs == 0 or epoch == args.num_epochs - 1:
                # save the model
                unet = accelerator.unwrap_model(model)

                if args.use_ema:
                    ema_model.store(unet.parameters())
                    ema_model.copy_to(unet.parameters())
                if args.with_vae:
                    pipeline = LDMPipelineRange(
                        unet=unet,
                        scheduler=noise_scheduler,
                        vae=vae,
                        pos_encoding=hasattr(args, 'pos_encoding') and args.pos_encoding,
                    )
                else:
                    pipeline = DDIMPipelineRange(
                        unet=unet,
                        scheduler=noise_scheduler,
                        pos_encoding=hasattr(args, 'pos_encoding') and args.pos_encoding,
                    ) 

                pipeline.save_pretrained(args.output_dir)

                if args.use_ema:
                    ema_model.restore(unet.parameters())

                if args.push_to_hub:
                    repo.push_to_hub(commit_message=f"Epoch {epoch}", blocking=False)
                del pipeline

    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)
