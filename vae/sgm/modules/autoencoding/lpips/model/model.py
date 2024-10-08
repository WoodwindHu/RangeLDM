import functools

import torch.nn as nn
import torch

from ..util import ActNorm


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator as in Pix2Pix
    --> see https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py
    """

    def __init__(self, input_nc=3, ndf=64, n_layers=3, use_actnorm=False):
        """Construct a PatchGAN discriminator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()
        if not use_actnorm:
            norm_layer = nn.BatchNorm2d
        else:
            norm_layer = ActNorm
        if (
            type(norm_layer) == functools.partial
        ):  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func != nn.BatchNorm2d
        else:
            use_bias = norm_layer != nn.BatchNorm2d

        kw = 4
        padw = 1
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True),
        ]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                nn.Conv2d(
                    ndf * nf_mult_prev,
                    ndf * nf_mult,
                    kernel_size=kw,
                    stride=2,
                    padding=padw,
                    bias=use_bias,
                ),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True),
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [
            nn.Conv2d(
                ndf * nf_mult_prev,
                ndf * nf_mult,
                kernel_size=kw,
                stride=1,
                padding=padw,
                bias=use_bias,
            ),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True),
        ]

        sequence += [
            nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)
        ]  # output 1 channel prediction map
        self.main = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.main(input)

class MetaKernel(nn.Module):
    def __init__(self, in_channels, out_channels, azi, inc, kernel_size=4, stride=2, padding=1):
        super().__init__()
        self.mlp_coord = nn.Sequential(
            nn.Linear(3, in_channels),
            nn.LeakyReLU(0.2, True),
            nn.Linear(in_channels, in_channels),
        )
        azi = torch.Tensor([azi])
        inc = torch.Tensor([inc])

        self.coov = nn.Conv2d(kernel_size*kernel_size*in_channels, out_channels, 1, 1, 0)
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        cos_azi = torch.zeros(kernel_size, kernel_size)
        sin_azi = torch.zeros(kernel_size, kernel_size)
        cos_inc = torch.zeros(kernel_size, kernel_size)
        sin_inc = torch.zeros(kernel_size, kernel_size)
        for shift_h in range(kernel_size):
            for shift_w in range(kernel_size):
                cos_azi[shift_h, shift_w] = torch.cos(azi*(shift_w-kernel_size//2))
                sin_azi[shift_h, shift_w] = torch.sin(azi*(shift_w-kernel_size//2))
                cos_inc[shift_h, shift_w] = torch.cos(inc*(shift_h-kernel_size//2))
                sin_inc[shift_h, shift_w] = torch.sin(inc*(shift_h-kernel_size//2))
        
        cos_azi = cos_azi.reshape(1, 1, 1, 1, kernel_size, kernel_size)
        sin_azi = sin_azi.reshape(1, 1, 1, 1, kernel_size, kernel_size)
        cos_inc = cos_inc.reshape(1, 1, 1, 1, kernel_size, kernel_size)
        sin_inc = sin_inc.reshape(1, 1, 1, 1, kernel_size, kernel_size)
        self.register_buffer("cos_azi", cos_azi)
        self.register_buffer("sin_azi", sin_azi)
        self.register_buffer("cos_inc", cos_inc)
        self.register_buffer("sin_inc", sin_inc)


    
    def forward(self, x, r):
        '''
        :param x: input feature map [B, C, W, H]
        :param r: range image [B, 1, W, H]
        :output: output feature map
        '''
        B, C, W, H = x.shape
        device = x.device
        r = nn.functional.pad(r, (self.padding, self.padding, 0, 0), value=100.)
        r = nn.functional.pad(r, (0, 0, self.padding, self.padding), mode="circular")
        r_patches = r.unfold(3, self.kernel_size, self.stride).unfold(2, self.kernel_size, self.stride) # (B, 1, W//stride, H//stride, kernel_size, kernel_size)
        r_center = r_patches[:, :, :, :, self.kernel_size//2, self.kernel_size//2]
        ## positional encoding
        pe0 = r_patches*self.cos_azi*self.cos_inc - r_center.unsqueeze(4).unsqueeze(4)
        pe1 = r_patches*self.cos_azi*self.sin_inc
        pe2 = r_patches*self.sin_azi
        pe = torch.cat([pe0, pe1, pe2], dim=1).permute(0, 2, 3, 4, 5, 1)
        weights = self.mlp_coord(pe).permute(0, 5, 1, 2, 3, 4) # (B, C, W//stride, H//stride, kernel_size, kernel_size)
        x = nn.functional.pad(x, (self.padding, self.padding, 0, 0))
        x = nn.functional.pad(x, (0, 0, self.padding, self.padding), mode="circular")
        x_patches = x.unfold(3, self.kernel_size, self.stride).unfold(2, self.kernel_size, self.stride) # (B, C, W//stride, H//stride, kernel_size, kernel_size)
        x_patches = weights*x_patches
        W_out, H_out = x_patches.shape[2:4]
        x_patches = x_patches.permute(0, 1, 4, 5, 2, 3).reshape(B, C*self.kernel_size*self.kernel_size,  W_out, H_out)
        output = self.coov(x_patches)
        return output, r_center
    
class MetaKernelSequential(nn.Sequential):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def forward(
        self,
        x,
        r,
    ):
        for layer in self:
            if isinstance(layer, MetaKernel):
                x, r = layer(x, r)
            else:
                x = layer(x)
        return x, r
    
class NLayerDiscriminatorMetaKernel(nn.Module):
    def __init__(self, input_nc=2, 
                 ndf=64, 
                 n_layers=3, 
                 use_actnorm=False,  
                 azi=0.00613592, 
                 inc=0.0074594, 
                 log=False,
                 range_mean = 20.,
                 range_std = 40.):
        """Construct a PatchGAN discriminator
        Parameters:
            input_nc (int)  -- the number of channels in input range images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super().__init__()
        self.range_mean = range_mean
        self.range_std = range_std
        self.log = log
        if not use_actnorm:
            norm_layer = nn.BatchNorm2d
        else:
            norm_layer = ActNorm
        if (
            type(norm_layer) == functools.partial
        ):  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func != nn.BatchNorm2d
        else:
            use_bias = norm_layer != nn.BatchNorm2d

        kw = 4
        padw = 1
        sequence = [
            MetaKernel(input_nc, ndf, kernel_size=kw, stride=2, padding=padw, azi=azi, inc=inc),
            nn.LeakyReLU(0.2, True),
        ]
        azi *= 2
        inc *= 2
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                MetaKernel(
                    ndf * nf_mult_prev,
                    ndf * nf_mult,
                    kernel_size=kw,
                    stride=2,
                    padding=padw,
                    azi=azi, 
                    inc=inc
                ),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True),
            ]
            azi *= 2
            inc *= 2

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [
            MetaKernel(
                ndf * nf_mult_prev,
                ndf * nf_mult,
                kernel_size=kw,
                stride=1,
                padding=padw,
                azi=azi, 
                inc=inc
            ),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True),
        ]

        sequence += [
            MetaKernel(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw, azi=azi, inc=inc)
        ]  # output 1 channel prediction map
        self.main = MetaKernelSequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        r = input[:, :1]
        if self.log:
            r = r.clamp(0, 1.2)
            r = (64**r-1)/10
        else:
            r = (r*self.range_std+self.range_mean)/10
        x = input
        x, r = self.main(x, r)
        return x
        

class NLayerDiscriminatorMetaKernel2(nn.Module):
    def __init__(self, input_nc=2, 
                 ndf=64, 
                 n_layers=3, 
                 use_actnorm=False,  
                 azi=0.00613592, 
                 inc=0.0074594, 
                 log=False,
                 range_mean = 20.,
                 range_std = 40.):
        """Construct a PatchGAN discriminator
        Parameters:
            input_nc (int)  -- the number of channels in input range images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super().__init__()
        self.range_mean = range_mean
        self.range_std = range_std
        self.log = log
        if not use_actnorm:
            norm_layer = nn.BatchNorm2d
        else:
            norm_layer = ActNorm
        if (
            type(norm_layer) == functools.partial
        ):  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func != nn.BatchNorm2d
        else:
            use_bias = norm_layer != nn.BatchNorm2d

        kw = 4
        padw = 1
        sequence = [
            MetaKernel(input_nc, ndf, kernel_size=kw, stride=2, padding=padw, azi=azi, inc=inc),
            nn.LeakyReLU(0.2, True),
        ]
        azi *= 2
        inc *= 2
        nf_mult = 1
        nf_mult_prev = 1
        nf_mult_prev = nf_mult
        nf_mult = min(2, 8)
        sequence += [
            MetaKernel(
                ndf * nf_mult_prev,
                ndf * nf_mult,
                kernel_size=kw,
                stride=2,
                padding=padw,
                azi=azi, 
                inc=inc
            ),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True),
        ]
        azi *= 2
        inc *= 2
        for n in range(2, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                nn.Conv2d(
                    ndf * nf_mult_prev,
                    ndf * nf_mult,
                    kernel_size=kw,
                    stride=2,
                    padding=padw,
                    bias=use_bias,
                ),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True),
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [
            nn.Conv2d(
                ndf * nf_mult_prev,
                ndf * nf_mult,
                kernel_size=kw,
                stride=1,
                padding=padw,
                bias=use_bias,
            ),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True),
        ]

        sequence += [
            nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)
        ]  # output 1 channel prediction map
        self.main = MetaKernelSequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        r = input[:, :1]
        if self.log:
            r = r.clamp(0, 1.2)
            r = (64**r-1)/10
        else:
            r = (r*self.range_std+self.range_mean)/10
        x = input
        x, r = self.main(x, r)
        return x
        