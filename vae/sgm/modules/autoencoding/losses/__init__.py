from typing import Any, Union
from sgm.data.kitti360_range_image import point_cloud_to_range_image_KITTI
# from sgm.data.stf_range_image import range_image_to_point_cloud_stf_torch

import torch
import torch.nn as nn
from einops import rearrange

from ....util import default, instantiate_from_config
from ..lpips.loss.lpips import LPIPS
from ..lpips.model.model import NLayerDiscriminator, weights_init, NLayerDiscriminatorMetaKernel, NLayerDiscriminatorMetaKernel2
from ..lpips.vqperceptual import hinge_d_loss, vanilla_d_loss


def adopt_weight(weight, global_step, threshold=0, value=0.0):
    if global_step < threshold:
        weight = value
    return weight


class LatentLPIPS(nn.Module):
    def __init__(
        self,
        decoder_config,
        perceptual_weight=1.0,
        latent_weight=1.0,
        scale_input_to_tgt_size=False,
        scale_tgt_to_input_size=False,
        perceptual_weight_on_inputs=0.0,
    ):
        super().__init__()
        self.scale_input_to_tgt_size = scale_input_to_tgt_size
        self.scale_tgt_to_input_size = scale_tgt_to_input_size
        self.init_decoder(decoder_config)
        self.perceptual_loss = LPIPS().eval()
        self.perceptual_weight = perceptual_weight
        self.latent_weight = latent_weight
        self.perceptual_weight_on_inputs = perceptual_weight_on_inputs

    def init_decoder(self, config):
        self.decoder = instantiate_from_config(config)
        if hasattr(self.decoder, "encoder"):
            del self.decoder.encoder

    def forward(self, latent_inputs, latent_predictions, image_inputs, split="train"):
        log = dict()
        loss = (latent_inputs - latent_predictions) ** 2
        log[f"{split}/latent_l2_loss"] = loss.mean().detach()
        image_reconstructions = None
        if self.perceptual_weight > 0.0:
            image_reconstructions = self.decoder.decode(latent_predictions)
            image_targets = self.decoder.decode(latent_inputs)
            perceptual_loss = self.perceptual_loss(
                image_targets.contiguous(), image_reconstructions.contiguous()
            )
            loss = (
                self.latent_weight * loss.mean()
                + self.perceptual_weight * perceptual_loss.mean()
            )
            log[f"{split}/perceptual_loss"] = perceptual_loss.mean().detach()

        if self.perceptual_weight_on_inputs > 0.0:
            image_reconstructions = default(
                image_reconstructions, self.decoder.decode(latent_predictions)
            )
            if self.scale_input_to_tgt_size:
                image_inputs = torch.nn.functional.interpolate(
                    image_inputs,
                    image_reconstructions.shape[2:],
                    mode="bicubic",
                    antialias=True,
                )
            elif self.scale_tgt_to_input_size:
                image_reconstructions = torch.nn.functional.interpolate(
                    image_reconstructions,
                    image_inputs.shape[2:],
                    mode="bicubic",
                    antialias=True,
                )

            perceptual_loss2 = self.perceptual_loss(
                image_inputs.contiguous(), image_reconstructions.contiguous()
            )
            loss = loss + self.perceptual_weight_on_inputs * perceptual_loss2.mean()
            log[f"{split}/perceptual_loss_on_inputs"] = perceptual_loss2.mean().detach()
        return loss, log


class GeneralLPIPSWithDiscriminator(nn.Module):
    def __init__(
        self,
        disc_start: int,
        logvar_init: float = 0.0,
        pixelloss_weight=1.0,
        disc_num_layers: int = 3,
        disc_in_channels: int = 3,
        disc_factor: float = 1.0,
        disc_weight: float = 1.0,
        perceptual_weight: float = 1.0,
        disc_loss: str = "hinge",
        scale_input_to_tgt_size: bool = False,
        dims: int = 2,
        learn_logvar: bool = False,
        regularization_weights: Union[None, dict] = None,
        intensity_weight = 10.0, 
        perceptual_loss = None,
        used_feature = 2,
        range_weight = 40.0, 
        kitti = False,
        disc_bev = False,
        bev_perceptual = False,
        bev_rec_weight = 0.0,
        to_range_image = None,
        use_rec_loss_true = False,
        use_rec_loss_true_power = False,
        rec_power = 2,
        metakernel = 0,
        disc_ndf=64,
        darknet=False,
        wo_perceptual=False,
    ):
        super().__init__()
        self.dims = dims
        if self.dims > 2:
            print(
                f"running with dims={dims}. This means that for perceptual loss calculation, "
                f"the LPIPS loss will be applied to each frame independently. "
            )
        self.scale_input_to_tgt_size = scale_input_to_tgt_size
        assert disc_loss in ["hinge", "vanilla"]
        self.pixel_weight = pixelloss_weight
        self.darknet = darknet
        if self.darknet:
            assert perceptual_loss is not None
        self.wo_perceptual = wo_perceptual
        if wo_perceptual:
            self.perceptual_loss = None
            assert perceptual_weight == 0.0
        else:
            self.perceptual_loss = instantiate_from_config(perceptual_loss).eval() if perceptual_loss else LPIPS().eval()
        self.perceptual_weight = perceptual_weight
        # output log variance
        self.logvar = nn.Parameter(torch.ones(size=()) * logvar_init)
        self.learn_logvar = learn_logvar

        
        self.metakernel = metakernel
        self.disc_bev = disc_bev
        self.discriminator_iter_start = disc_start
        self.disc_loss = hinge_d_loss if disc_loss == "hinge" else vanilla_d_loss
        self.disc_factor = disc_factor
        self.discriminator_weight = disc_weight
        self.regularization_weights = default(regularization_weights, {})
        self.intensity_weight = intensity_weight
        self.used_feature = used_feature
        self.range_weight = range_weight
        self.kitti = kitti
        self.bev_perceptual = bev_perceptual
        self.bev_rec_weight = bev_rec_weight
        self.use_rec_loss_true = use_rec_loss_true
        self.use_rec_loss_true_power = use_rec_loss_true_power
        self.rec_power = rec_power
        if self.kitti:
            self.to_range_image = instantiate_from_config(to_range_image) if to_range_image else point_cloud_to_range_image_KITTI()
        else:
            raise NotImplementedError
        if metakernel == 1:
            self.discriminator = NLayerDiscriminatorMetaKernel(
                input_nc=disc_in_channels, 
                ndf=disc_ndf, 
                n_layers=disc_num_layers, 
                use_actnorm=False, 
                log=self.to_range_image.log, 
                range_mean=self.to_range_image.mean,
                range_std=self.to_range_image.std
            ).apply(weights_init)
        elif metakernel == 2:
            self.discriminator = NLayerDiscriminatorMetaKernel2(
                input_nc=disc_in_channels, 
                ndf=disc_ndf, 
                n_layers=disc_num_layers, 
                use_actnorm=False, 
                log=self.to_range_image.log, 
                range_mean=self.to_range_image.mean,
                range_std=self.to_range_image.std
            ).apply(weights_init)
        else:
            self.discriminator = NLayerDiscriminator(
                input_nc=disc_in_channels, n_layers=disc_num_layers, use_actnorm=False
            ).apply(weights_init) 

    def get_trainable_parameters(self) -> Any:
        return self.discriminator.parameters()

    def get_trainable_autoencoder_parameters(self) -> Any:
        if self.learn_logvar:
            yield self.logvar
        yield from ()

    def calculate_adaptive_weight(self, nll_loss, g_loss, last_layer=None):
        if last_layer is not None:
            nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]
        else:
            nll_grads = torch.autograd.grad(
                nll_loss, self.last_layer[0], retain_graph=True
            )[0]
            g_grads = torch.autograd.grad(
                g_loss, self.last_layer[0], retain_graph=True
            )[0]

        d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
        d_weight = d_weight * self.discriminator_weight
        return d_weight

    def forward(
        self,
        regularization_log,
        inputs,
        reconstructions,
        optimizer_idx,
        global_step,
        last_layer=None,
        split="train",
        weights=None,
    ):
        if self.scale_input_to_tgt_size:
            inputs = torch.nn.functional.interpolate(
                inputs, reconstructions.shape[2:], mode="bicubic", antialias=True
            )

        if self.dims > 2:
            inputs, reconstructions = map(
                lambda x: rearrange(x, "b c t h w -> (b t) c h w"),
                (inputs, reconstructions),
            )

        if self.to_range_image.log:
            rec_loss_true = torch.abs(64**inputs[:,0,:,:].contiguous() - 64**reconstructions[:,0,:,:].contiguous())
        elif self.to_range_image.inverse:
            rec_loss_true = torch.abs(1/torch.max(inputs[:,0,:,:].contiguous(),torch.tensor([0.0001], device=inputs.device)) - 1/torch.max(reconstructions[:,0,:,:].contiguous(),torch.tensor([0.0001], device=inputs.device)))
        if self.use_rec_loss_true:
            if not self.to_range_image.log and not self.to_range_image.inverse:
                raise NotImplementedError
            rec_loss = rec_loss_true
        elif self.use_rec_loss_true_power:
            if not self.to_range_image.log:
                raise NotImplementedError
            rec_loss = torch.abs(((64**inputs[:,0,:,:])**self.rec_power).contiguous() - ((64**reconstructions[:,0,:,:])**self.rec_power).contiguous())
        else:
            rec_loss = self.range_weight * torch.abs(inputs[:,0,:,:].contiguous() - reconstructions[:,0,:,:].contiguous()) 
        if self.used_feature > 1:
            rec_loss += self.intensity_weight * torch.abs(inputs[:,1,:,:].contiguous() - reconstructions[:,1,:,:].contiguous())
        is_voxel = False
            
        if self.perceptual_weight > 0:
            assert self.kitti == True
            if self.darknet:
                inputs_dark = self.to_range_image.with_xyz(inputs.contiguous())
                reconstructions_dark = self.to_range_image.with_xyz(reconstructions.contiguous())
                p_loss = self.perceptual_loss(
                    inputs_dark, 
                    reconstructions_dark, 
                )
                rec_loss = rec_loss + self.perceptual_weight * p_loss
            elif self.bev_perceptual and not is_voxel:
                is_voxel = True 
                inputs = self.to_range_image.to_voxel(inputs.contiguous())
                reconstructions = self.to_range_image.to_voxel(reconstructions.contiguous())
                p_loss = self.perceptual_loss(
                    torch.cat((inputs[:,:1,:,:], inputs[:,:1,:,:], inputs[:,1:,:,:]), dim=1), 
                    torch.cat((reconstructions[:,:1,:,:], reconstructions[:,:1,:,:], reconstructions[:,1:,:,:]), dim=1), 
                )
                rec_loss = rec_loss + self.perceptual_weight * p_loss
            # p_loss = self.perceptual_loss(
            #     inputs[:,:1,:,:].contiguous().repeat(1,3,1,1), reconstructions[:,:1,:,:].contiguous().repeat(1,3,1,1)
            # ) + self.perceptual_loss(
            #     inputs[:,1:,:,:].contiguous().repeat(1,3,1,1), reconstructions[:,1:,:,:].contiguous().repeat(1,3,1,1)
            # ) 
            # rec_loss = rec_loss + self.perceptual_weight * p_loss
            else:
                inputs_pc = self.to_range_image.to_pc_torch(inputs.contiguous())
                reconstructions_pc = self.to_range_image.to_pc_torch(reconstructions.contiguous())
                p_loss = self.perceptual_loss(
                    inputs_pc.contiguous(), reconstructions_pc.contiguous()
                )
                # for p_loss.shape = [B, 1, point_number]
                if p_loss.shape[-1] != 1:
                    p_loss = p_loss.view(inputs.shape[0], inputs.shape[2], inputs.shape[3])
                rec_loss = rec_loss + self.perceptual_weight * p_loss

        nll_loss = rec_loss / torch.exp(self.logvar) + self.logvar
        weighted_nll_loss = nll_loss
        if weights is not None:
            weighted_nll_loss = weights * nll_loss
        weighted_nll_loss = torch.sum(weighted_nll_loss) / weighted_nll_loss.shape[0]
        nll_loss = torch.sum(nll_loss) / nll_loss.shape[0]
        if self.bev_rec_weight > 0:
            if not is_voxel:
                is_voxel = True 
                inputs = self.to_range_image.to_voxel(inputs.contiguous())
                reconstructions = self.to_range_image.to_voxel(reconstructions.contiguous())
            bev_rec_loss = self.bev_rec_weight * torch.abs(inputs[:,0,:,:].contiguous() - reconstructions[:,0,:,:].contiguous())
            nll_loss += torch.sum(bev_rec_loss) / bev_rec_loss.shape[0]
        
        # now the GAN part
        if optimizer_idx == 0:
            if self.disc_bev and not is_voxel:
                # inputs = self.to_range_image.to_voxel(inputs.contiguous())
                reconstructions = self.to_range_image.to_voxel(reconstructions.contiguous())
            # generator update
            logits_fake = self.discriminator(reconstructions.contiguous())
            g_loss = -torch.mean(logits_fake)
            if self.disc_factor > 0.0:
                try:
                    d_weight = self.calculate_adaptive_weight(
                        nll_loss, g_loss, last_layer=last_layer
                    )
                except RuntimeError:
                    assert not self.training
                    d_weight = torch.tensor(0.0)
            else:
                d_weight = torch.tensor(0.0)

            disc_factor = adopt_weight(
                self.disc_factor, global_step, threshold=self.discriminator_iter_start
            )
            loss = weighted_nll_loss + d_weight * disc_factor * g_loss
            log = dict()
            for k in regularization_log:
                if k in self.regularization_weights:
                    loss = loss + self.regularization_weights[k] * regularization_log[k]
                log[f"{split}/{k}"] = regularization_log[k].detach().mean()

            log.update(
                {
                    "{}/total_loss".format(split): loss.clone().detach().mean(),
                    "{}/logvar".format(split): self.logvar.detach(),
                    "{}/nll_loss".format(split): nll_loss.detach().mean(),
                    "{}/rec_loss".format(split): rec_loss.detach().mean(),
                    "{}/d_weight".format(split): d_weight.detach(),
                    "{}/disc_factor".format(split): torch.tensor(disc_factor),
                    "{}/g_loss".format(split): g_loss.detach().mean(),
                }
            )

            if self.perceptual_weight > 0:
                log.update({"{}/p_loss".format(split): p_loss.detach().mean()})

            if self.to_range_image.log or self.to_range_image.inverse:
                log.update({"{}/rec_loss_true".format(split): rec_loss_true.detach().mean()})
            
            if self.bev_rec_weight>0:
                log.update({"{}/bev_rec_loss".format(split): bev_rec_loss.detach().mean()})

            return loss, log

        if optimizer_idx == 1:
            if self.disc_bev and not is_voxel:
                inputs = self.to_range_image.to_voxel(inputs.contiguous())
                reconstructions = self.to_range_image.to_voxel(reconstructions.contiguous())

            # second pass for discriminator update
            logits_real = self.discriminator(inputs.contiguous().detach())
            logits_fake = self.discriminator(reconstructions.contiguous().detach())

            disc_factor = adopt_weight(
                self.disc_factor, global_step, threshold=self.discriminator_iter_start
            )
            d_loss = disc_factor * self.disc_loss(logits_real, logits_fake)

            log = {
                "{}/disc_loss".format(split): d_loss.clone().detach().mean(),
                "{}/logits_real".format(split): logits_real.detach().mean(),
                "{}/logits_fake".format(split): logits_fake.detach().mean(),
            }
            return d_loss, log

class RangeImageReconstructionLoss(nn.Module):
    def __init__(
        self,
        intensity_weight = 10.0,
        logvar_init: float = 0.0,
        regularization_weights: Union[None, dict] = None,
        kitti: bool = False,
        used_feature: int = 2,
        learn_logvar: bool = False,
        perceptual_loss = None,
        perceptual_weight = 0.0,
        range_weight = 40.0,
    ):
        super().__init__()
        self.logvar = nn.Parameter(torch.ones(size=()) * logvar_init)
        self.regularization_weights = default(regularization_weights, {})
        self.perceptual_loss = instantiate_from_config(perceptual_loss).eval() if perceptual_loss else None
        self.perceptual_weight = perceptual_weight
        self.intensity_weight = intensity_weight
        self.range_weight = range_weight
        
        self.kitti = kitti
        if self.kitti:
            self.to_range_image = point_cloud_to_range_image_KITTI()
        self.used_feature = used_feature
        self.learn_logvar = learn_logvar

    def get_trainable_parameters(self) -> Any:
        yield from ()

    def get_trainable_autoencoder_parameters(self) -> Any:
        if self.learn_logvar:
            yield self.logvar
        yield from ()

    def forward(
        self,
        regularization_log,
        inputs,
        reconstructions,
        global_step,
        last_layer=None,
        split="train",
        weights=None,
    ):
        rec_loss = self.range_weight * torch.abs(inputs[:,0,:,:].contiguous() - reconstructions[:,0,:,:].contiguous()) 
        if self.used_feature > 1:
            rec_loss += self.intensity_weight * torch.abs(inputs[:,1,:,:].contiguous() - reconstructions[:,1,:,:].contiguous())

        if self.perceptual_weight > 0:
            assert self.kitti == True
            inputs_pc = self.to_range_image.to_pc_torch(inputs.contiguous())
            reconstructions_pc = self.to_range_image.to_pc_torch(reconstructions.contiguous())
            p_loss = self.perceptual_loss(
                inputs_pc.contiguous(), reconstructions_pc.contiguous()
            )
            # for p_loss.shape = [B, 1, point_number]
            if p_loss.shape[-1] != 1:
                p_loss = p_loss.view(inputs.shape[0], inputs.shape[2], inputs.shape[3])
            rec_loss = rec_loss + self.perceptual_weight * p_loss

        nll_loss = rec_loss / torch.exp(self.logvar) + self.logvar
        weighted_nll_loss = nll_loss
        if weights is not None:
            weighted_nll_loss = weights * nll_loss
        loss = torch.sum(weighted_nll_loss) / weighted_nll_loss.shape[0]
        nll_loss = torch.sum(nll_loss) / nll_loss.shape[0]

        log = dict()
        for k in regularization_log:
                if k in self.regularization_weights:
                    loss = loss + self.regularization_weights[k] * regularization_log[k]
                log[f"{split}/{k}"] = regularization_log[k].detach().mean()
        log.update(
                {
                    "{}/total_loss".format(split): loss.clone().detach().mean(),
                    "{}/logvar".format(split): self.logvar.detach(),
                    "{}/nll_loss".format(split): nll_loss.detach().mean(),
                    "{}/p_loss".format(split): p_loss.detach().mean() if 'p_loss' in dir() else 0,
                    "{}/rec_loss".format(split): rec_loss.detach().mean(),
                }
            )

        return loss, log