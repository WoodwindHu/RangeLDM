# pytorch_diffusion + derived encoder decoder
import math
from typing import Any, Callable, Optional, Union

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from packaging import version
import math
from functools import partial
from torch.nn.common_types import _size_2_t
import torch.nn.functional as F
from torch.nn.modules.utils import _pair

try:
    import xformers
    import xformers.ops

    XFORMERS_IS_AVAILABLE = True
except:
    XFORMERS_IS_AVAILABLE = False
    print("no module 'xformers'. Processing without...")

from ...modules.attention import LinearAttention, MemoryEfficientCrossAttention


def get_timestep_embedding(timesteps, embedding_dim):
    """
    This matches the implementation in Denoising Diffusion Probabilistic Models:
    From Fairseq.
    Build sinusoidal embeddings.
    This matches the implementation in tensor2tensor, but differs slightly
    from the description in Section 3.5 of "Attention Is All You Need".
    """
    assert len(timesteps.shape) == 1

    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
    emb = emb.to(device=timesteps.device)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb


def nonlinearity(x, kind='relu'):
    # swish
    if kind == 'silu':
        return x * torch.sigmoid(x)
    elif kind == 'relu':
        return torch.nn.functional.relu(x)
    else:
        raise NotImplementedError


def Normalize(in_channels, num_groups=32):
    return torch.nn.GroupNorm(
        num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True
    )

class Conv2d(torch.nn.Conv2d):
    def __init__(self, 
                 in_channels: int,
                 out_channels: int,
                 kernel_size: _size_2_t,
                 stride: _size_2_t = 1,
                 padding: Union[str, _size_2_t] = 0,
                 dilation: _size_2_t = 1,
                 groups: int = 1,
                 bias: bool = True,
                 padding_mode: str = 'zeros',  # TODO: refine this type
                 device=None,
                 dtype=None,
                 circular=False,
                 coord=False):
        super().__init__(in_channels=in_channels if not coord else in_channels+1, 
                         out_channels=out_channels, 
                         kernel_size=kernel_size, 
                         stride=stride, 
                         padding=padding,
                         dilation=dilation,
                         groups=groups,
                         bias=bias,
                         padding_mode=padding_mode,  
                         device=device,
                         dtype=dtype,)
        self.circular = circular
        self.coord = coord

    def _conv_forward(self, input: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor]):
        if self.coord:
            # coordconv 
            B, C, W, H = input.shape
            y_coords = torch.linspace(-1, 1, H, device=input.device)
            input = torch.cat([input, y_coords[None, None, None, :].repeat(B, 1, W, 1)], dim=1)
        if self.circular:
            input = F.pad(input, (0, 0, self.padding[0], self.padding[0]), mode="circular")
            input = F.pad(input, (self.padding[1], self.padding[1], 0, 0), mode="constant")
            return F.conv2d(input, weight, bias, self.stride, _pair(0), self.dilation, self.groups)
        if self.padding_mode != 'zeros':
            return F.conv2d(F.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode),
                            weight, bias, self.stride,
                            _pair(0), self.dilation, self.groups)
        return F.conv2d(input, weight, bias, self.stride,
                        self.padding, self.dilation, self.groups)

class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv, up_single, circular=False, coord=False):
        super().__init__()
        self.with_conv = with_conv
        self.up_single = up_single
        if self.with_conv:
            self.conv = Conv2d(
                in_channels, in_channels, kernel_size=3, stride=1, padding=1, circular=circular, coord=coord
            )

    def forward(self, x):
        x = torch.nn.functional.interpolate(x, scale_factor=2.0 if not self.up_single else (2.0, 1.0),
                                            mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x

class RangeDownSample(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x, r):
        '''
        :param x: B*C*W*H
        :param r: B*1*W*H
        '''
        B, C, W, H = x.shape
        # unfold x to 2x2 blocks
        x_unfolded = x.unfold(2, 2, 2).unfold(3, 2, 2) # B*C*W/2*H/2*2*2
        r_unfolded = r.unfold(2, 2, 2).unfold(3, 2, 2) # B*1*W/2*H/2*2*2

        x_reshaped = x_unfolded.reshape(B, C, W//2, H//2, 4)
        r_reshaped = r_unfolded.reshape(B, 1, W//2, H//2, 4)
        r_mean = r_reshaped.mean(dim=-1, keepdim=True)
        idx = ((r_reshaped - r_mean)**2).argmin(-1, keepdim=True)
        r_out = torch.gather(r_reshaped, dim=-1, index=idx).squeeze(-1)
        x_out = torch.gather(x_reshaped, dim=-1, index=idx.repeat(1, C, 1, 1, 1)).squeeze(-1)

        return x_out, r_out
                

class Downsample(nn.Module):
    def __init__(self, in_channels, with_conv, down_single, circular=False, coord=False):
        super().__init__()
        self.with_conv = with_conv
        self.down_single = down_single
        self.circular = circular
        if self.with_conv:
            # no asymmetric padding in torch conv, must do it ourselves
            self.conv = Conv2d(
                in_channels, in_channels, kernel_size=3, stride=2 if not self.down_single else (2, 1), 
                padding=0, coord=coord
            )

    def forward(self, x):
        if self.with_conv:
            if not self.circular:
                pad = (0, 1, 0, 1) if not self.down_single else (1, 1, 0, 1)
                x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
            else:
                x = torch.nn.functional.pad(x, (0, 0, 0, 1), mode="circular")
                x = torch.nn.functional.pad(x, (1 if self.down_single else 0, 1, 0, 0), mode="constant")
            x = self.conv(x)
        else:
            x = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
        return x

class EdgeConv(nn.Module):
    def __init__(self, in_channels, out_channels, azi, inc, act='relu'):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Conv2d(2*in_channels+3, out_channels, 1, 1, 0),
            nn.ReLU() if act == 'relu' else nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, 1, 1, 0),
        )
        self.azi = azi
        self.inc = inc 
    
    def forward(self, x, r):
        '''
        :param x: input feature map [B, C, W, H]
        :param r: range image [B, 1, W, H]
        :output: output feature map
        '''
        B, C, W, H = x.shape
        output_features = []
        for shift_h in range(-1, 2):
            for shift_w in range(-1, 2):
                x_shift = torch.roll(x, shifts=(shift_w, shift_h), dims=(2, 3))
                r_shift = torch.roll(r, shifts=(shift_w, shift_h), dims=(2, 3))
                
                # # shift x
                # x_shift_tmp = x.clone()
                # r_shift_tmp = r.clone()
                # # cicular padding
                # if shift_w != 0:
                #     x_shift_tmp[:, :, shift_w:, :] = x[:, :, :-shift_w, :]
                #     x_shift_tmp[:, :, :shift_w, :] = x[:, :, -shift_w:, :]
                #     r_shift_tmp[:, :, shift_w:, :] = r[:, :, :-shift_w, :]
                #     r_shift_tmp[:, :, :shift_w, :] = r[:, :, -shift_w:, :]
                # x_shift = x_shift_tmp.clone()
                # r_shift = r_shift_tmp.clone()
                # # zero padding
                # if shift_h > 0:
                #     x_shift[:, :, :, shift_h:] = x_shift_tmp[:, :, :, :-shift_h]
                #     x_shift[:, :, :, 0:shift_h] = 0
                #     r_shift[:, :, :, shift_h:] = r_shift_tmp[:, :, :, :-shift_h]
                #     r_shift[:, :, :, 0:shift_h] = 2
                # elif shift_h < 0:
                #     x_shift[:, :, :, :shift_h] = x_shift_tmp[:, :, :, -shift_h:]
                #     x_shift[:, :, :, shift_h:] = 0
                #     r_shift[:, :, :, :shift_h] = r_shift_tmp[:, :, :, -shift_h:]
                #     r_shift[:, :, :, shift_h:] = 2
                ## positional encoding
                pe0 = r_shift*math.cos(shift_w*self.azi)*math.cos(shift_h*self.inc) - r 
                pe1 = r_shift*math.cos(shift_w*self.azi)*math.sin(shift_h*self.inc)
                pe2 = r_shift*math.sin(shift_w*self.azi)
                input_feature = torch.cat((x_shift, x, pe0, pe1, pe2), dim=1)
                output_feature = self.mlp(input_feature)
                output_features.append(output_feature)
        output, _ = torch.stack(output_features).max(dim=0)
        return output
                

class EdgeConvResnetBlock(nn.Module):
    def __init__(
        self,
        *,
        in_channels,
        azi,
        inc,
        out_channels=None,
        conv_shortcut=False,
        dropout,
        act = 'relu'
    ):
        '''
        :param azi: azimuth difference
        :param inc: inclination difference
        '''
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut
        self.nonlinearity = partial(nonlinearity, kind=act)

        self.norm1 = Normalize(in_channels)
        self.conv1 = EdgeConv(
            in_channels, out_channels, azi=azi, inc=inc, act=act
        )
        self.norm2 = Normalize(out_channels)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = EdgeConv(
            out_channels, out_channels, azi=azi, inc=inc, act=act
        )
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = EdgeConv(
                    in_channels, out_channels, azi=azi, inc=inc, act=act
                )
            else:
                self.nin_shortcut = torch.nn.Conv2d(
                    in_channels, out_channels, kernel_size=1, stride=1, padding=0
                )

    def forward(self, x, r):
        '''
        :param x: input feature map
        :param r: range image
        :output: output feature map
        '''
        h = x
        h = self.norm1(h)
        h = self.nonlinearity(h)
        h = self.conv1(h, r)


        h = self.norm2(h)
        h = self.nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h, r)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x, r)
            else:
                x = self.nin_shortcut(x)

        return x + h

class ResnetBlock(nn.Module):
    def __init__(
        self,
        *,
        in_channels,
        out_channels=None,
        conv_shortcut=False,
        dropout,
        temb_channels=512,
        act = 'relu',
        circular=False,
        coord=False
    ):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels)
        self.conv1 = Conv2d(
            in_channels, out_channels, kernel_size=3, stride=1, padding=1, circular=circular, coord=coord
        )
        if temb_channels > 0:
            self.temb_proj = torch.nn.Linear(temb_channels, out_channels)
        self.norm2 = Normalize(out_channels)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, circular=circular, coord=coord
        )
        self.nonlinearity = partial(nonlinearity, kind=act)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = Conv2d(
                    in_channels, out_channels, kernel_size=3, stride=1, padding=1, circular=circular, coord=coord
                )
            else:
                self.nin_shortcut = torch.nn.Conv2d(
                    in_channels, out_channels, kernel_size=1, stride=1, padding=0
                )

    def forward(self, x, temb):
        h = x
        h = self.norm1(h)
        h = self.nonlinearity(h)
        h = self.conv1(h)

        if temb is not None:
            h = h + self.temb_proj(self.nonlinearity(temb))[:, :, None, None]

        h = self.norm2(h)
        h = self.nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x + h


class LinAttnBlock(LinearAttention):
    """to match AttnBlock usage"""

    def __init__(self, in_channels):
        super().__init__(dim=in_channels, heads=1, dim_head=in_channels)


class AttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.k = torch.nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.v = torch.nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.proj_out = torch.nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )

    def attention(self, h_: torch.Tensor) -> torch.Tensor:
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        b, c, h, w = q.shape
        q, k, v = map(
            lambda x: rearrange(x, "b c h w -> b 1 (h w) c").contiguous(), (q, k, v)
        )
        h_ = torch.nn.functional.scaled_dot_product_attention(
            q, k, v
        )  # scale is dim ** -0.5 per default
        # compute attention

        return rearrange(h_, "b 1 (h w) c -> b c h w", h=h, w=w, c=c, b=b)

    def forward(self, x, **kwargs):
        h_ = x
        h_ = self.attention(h_)
        h_ = self.proj_out(h_)
        return x + h_


class MemoryEfficientAttnBlock(nn.Module):
    """
    Uses xformers efficient implementation,
    see https://github.com/MatthieuTPHR/diffusers/blob/d80b531ff8060ec1ea982b65a1b8df70f73aa67c/src/diffusers/models/attention.py#L223
    Note: this is a single-head self-attention operation
    """

    #
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.k = torch.nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.v = torch.nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.proj_out = torch.nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.attention_op: Optional[Any] = None

    def attention(self, h_: torch.Tensor) -> torch.Tensor:
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        B, C, H, W = q.shape
        q, k, v = map(lambda x: rearrange(x, "b c h w -> b (h w) c"), (q, k, v))

        q, k, v = map(
            lambda t: t.unsqueeze(3)
            .reshape(B, t.shape[1], 1, C)
            .permute(0, 2, 1, 3)
            .reshape(B * 1, t.shape[1], C)
            .contiguous(),
            (q, k, v),
        )
        out = xformers.ops.memory_efficient_attention(
            q, k, v, attn_bias=None, op=self.attention_op
        )

        out = (
            out.unsqueeze(0)
            .reshape(B, 1, out.shape[1], C)
            .permute(0, 2, 1, 3)
            .reshape(B, out.shape[1], C)
        )
        return rearrange(out, "b (h w) c -> b c h w", b=B, h=H, w=W, c=C)

    def forward(self, x, **kwargs):
        h_ = x
        h_ = self.attention(h_)
        h_ = self.proj_out(h_)
        return x + h_


class MemoryEfficientCrossAttentionWrapper(MemoryEfficientCrossAttention):
    def forward(self, x, context=None, mask=None, **unused_kwargs):
        b, c, h, w = x.shape
        x = rearrange(x, "b c h w -> b (h w) c")
        out = super().forward(x, context=context, mask=mask)
        out = rearrange(out, "b (h w) c -> b c h w", h=h, w=w, c=c)
        return x + out


def make_attn(in_channels, attn_type="vanilla", attn_kwargs=None):
    assert attn_type in [
        "vanilla",
        "vanilla-xformers",
        "memory-efficient-cross-attn",
        "linear",
        "none",
    ], f"attn_type {attn_type} unknown"
    if (
        version.parse(torch.__version__) < version.parse("2.0.0")
        and attn_type != "none"
    ):
        assert XFORMERS_IS_AVAILABLE, (
            f"We do not support vanilla attention in {torch.__version__} anymore, "
            f"as it is too expensive. Please install xformers via e.g. 'pip install xformers==0.0.16'"
        )
        attn_type = "vanilla-xformers"
    print(f"making attention of type '{attn_type}' with {in_channels} in_channels")
    if attn_type == "vanilla":
        assert attn_kwargs is None
        return AttnBlock(in_channels)
    elif attn_type == "vanilla-xformers":
        print(f"building MemoryEfficientAttnBlock with {in_channels} in_channels...")
        return MemoryEfficientAttnBlock(in_channels)
    elif type == "memory-efficient-cross-attn":
        attn_kwargs["query_dim"] = in_channels
        return MemoryEfficientCrossAttentionWrapper(**attn_kwargs)
    elif attn_type == "none":
        return nn.Identity(in_channels)
    else:
        return LinAttnBlock(in_channels)


class Model(nn.Module):
    def __init__(
        self,
        *,
        ch,
        out_ch,
        ch_mult=(1, 2, 4, 8),
        num_res_blocks,
        attn_resolutions,
        dropout=0.0,
        resamp_with_conv=True,
        in_channels,
        resolution,
        use_timestep=True,
        use_linear_attn=False,
        attn_type="vanilla",
        act='silu',
        circular=False,
    ):
        super().__init__()
        if use_linear_attn:
            attn_type = "linear"
        self.ch = ch
        self.temb_ch = self.ch * 4
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.nonlinearity = partial(nonlinearity, kind=act)

        self.use_timestep = use_timestep
        if self.use_timestep:
            # timestep embedding
            self.temb = nn.Module()
            self.temb.dense = nn.ModuleList(
                [
                    torch.nn.Linear(self.ch, self.temb_ch),
                    torch.nn.Linear(self.temb_ch, self.temb_ch),
                ]
            )

        # downsampling
        self.conv_in = Conv2d(
            in_channels, self.ch, kernel_size=3, stride=1, padding=1, circular=circular
        )

        curr_res = resolution
        in_ch_mult = (1,) + tuple(ch_mult)
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(
                    ResnetBlock(
                        in_channels=block_in,
                        out_channels=block_out,
                        temb_channels=self.temb_ch,
                        dropout=dropout,
                        act=act,
                        circular=circular,
                    )
                )
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(make_attn(block_in, attn_type=attn_type))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:
                down.downsample = Downsample(block_in, resamp_with_conv, circular=circular)
                curr_res = curr_res // 2
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout,
            act=act,
            circular=circular,
        )
        self.mid.attn_1 = make_attn(block_in, attn_type=attn_type)
        self.mid.block_2 = ResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout,
            act=act,
            circular=circular,
        )

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            skip_in = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks + 1):
                if i_block == self.num_res_blocks:
                    skip_in = ch * in_ch_mult[i_level]
                block.append(
                    ResnetBlock(
                        in_channels=block_in + skip_in,
                        out_channels=block_out,
                        temb_channels=self.temb_ch,
                        dropout=dropout,
                        act=act,
                        circular=circular,
                    )
                )
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(make_attn(block_in, attn_type=attn_type))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv, circular=circular)
                curr_res = curr_res * 2
            self.up.insert(0, up)  # prepend to get consistent order

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = Conv2d(
            block_in, out_ch, kernel_size=3, stride=1, padding=1, circular=circular
        )

    def forward(self, x, t=None, context=None):
        # assert x.shape[2] == x.shape[3] == self.resolution
        if context is not None:
            # assume aligned context, cat along channel axis
            x = torch.cat((x, context), dim=1)
        if self.use_timestep:
            # timestep embedding
            assert t is not None
            temb = get_timestep_embedding(t, self.ch)
            temb = self.temb.dense[0](temb)
            temb = self.nonlinearity(temb)
            temb = self.temb.dense[1](temb)
        else:
            temb = None

        # downsampling
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1], temb)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions - 1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        # middle
        h = hs[-1]
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](
                    torch.cat([h, hs.pop()], dim=1), temb
                )
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # end
        h = self.norm_out(h)
        h = self.nonlinearity(h)
        h = self.conv_out(h)
        return h

    def get_last_layer(self):
        return self.conv_out.weight


class Encoder(nn.Module):
    def __init__(
        self,
        *,
        ch,
        out_ch,
        ch_mult=(1, 2, 4, 8),
        num_res_blocks,
        attn_resolutions,
        dropout=0.0,
        resamp_with_conv=True,
        resamp_single_side=False,
        in_channels,
        resolution,
        z_channels,
        double_z=True,
        use_linear_attn=False,
        attn_type="vanilla",
        edge_conv=False,
        azi=0,
        inc=0,
        act='relu',
        circular=False,
        coord=False,
        **ignore_kwargs,
    ):
        super().__init__()
        if use_linear_attn:
            attn_type = "linear"
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.edge_conv = edge_conv
        self.nonlinearity = partial(nonlinearity, kind=act)

        # downsampling
        if self.edge_conv:
            self.conv_in = EdgeConv(
                in_channels, self.ch, azi=azi, inc=inc, act=act
            )
        else:
            self.conv_in = Conv2d(
                in_channels, self.ch, kernel_size=3, stride=1, padding=1, circular=circular, coord=coord
            )

        curr_res = resolution
        in_ch_mult = (1,) + tuple(ch_mult)
        self.in_ch_mult = in_ch_mult
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(
                    ResnetBlock(
                        in_channels=block_in,
                        out_channels=block_out,
                        temb_channels=self.temb_ch,
                        dropout=dropout,
                        act=act,
                        circular=circular,
                        coord=coord
                    ) if not self.edge_conv else EdgeConvResnetBlock(
                        in_channels=block_in,
                        out_channels=block_out,
                        azi = azi,
                        inc = inc,
                        dropout=dropout,
                        act=act
                    )
                )
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(make_attn(block_in, attn_type=attn_type))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:
                if not self.edge_conv:
                    down.downsample = Downsample(block_in, resamp_with_conv, resamp_single_side, circular=circular, coord=coord)  
                else:
                    down.downsample = RangeDownSample()
                curr_res = curr_res // 2
                azi *= 2
                inc *= 2
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout,
            act=act,
            circular=circular,
            coord=coord
        ) if not self.edge_conv else EdgeConvResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            azi = azi,
            inc = inc,
            dropout=dropout,
            act=act
        )
        self.mid.attn_1 = make_attn(block_in, attn_type=attn_type)
        self.mid.block_2 = ResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout,
            act=act,
            circular=circular,
            coord=coord
        ) if not self.edge_conv else EdgeConvResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            azi = azi,
            inc = inc,
            dropout=dropout,
            act=act
        )

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = Conv2d(
            block_in,
            2 * z_channels if double_z else z_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            circular=circular,
            coord=coord
        ) if not self.edge_conv else EdgeConv(
            block_in, 
            2 * z_channels if double_z else z_channels, 
            azi=azi, 
            inc=inc
        )

    def forward(self, x):
        # timestep embedding
        temb = None
        r = x[:,:1, :, :].clone().detach()
        # breakpoint()
        # downsampling
        if self.edge_conv:
            hs = [self.conv_in(x, r)]
        else:
            hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                if self.edge_conv:
                    h = self.down[i_level].block[i_block](hs[-1], r)
                else:
                    h = self.down[i_level].block[i_block](hs[-1], temb)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions - 1:
                if self.edge_conv:
                    h, r = self.down[i_level].downsample(hs[-1], r)
                    hs.append(h)
                else:
                    hs.append(self.down[i_level].downsample(hs[-1]))

        # middle
        h = hs[-1]
        if self.edge_conv:
            h = self.mid.block_1(h, r)
            h = self.mid.attn_1(h)
            h = self.mid.block_2(h, r)
        else:
            h = self.mid.block_1(h, temb)
            h = self.mid.attn_1(h)
            h = self.mid.block_2(h, temb)

        # end
        h = self.norm_out(h)
        h = self.nonlinearity(h)
        if self.edge_conv:
            h = self.conv_out(h, r)
        else:
            h = self.conv_out(h)
        return h


class Decoder(nn.Module):
    def __init__(
        self,
        *,
        ch,
        out_ch,
        ch_mult=(1, 2, 4, 8),
        num_res_blocks,
        attn_resolutions,
        dropout=0.0,
        resamp_with_conv=True,
        resamp_single_side=False,
        in_channels,
        resolution,
        z_channels,
        give_pre_end=False,
        tanh_out=False,
        use_linear_attn=False,
        attn_type="vanilla",
        act='relu',
        circular=False,
        coord=False,
        **ignorekwargs,
    ):
        super().__init__()
        if use_linear_attn:
            attn_type = "linear"
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.give_pre_end = give_pre_end
        self.tanh_out = tanh_out
        self.nonlinearity = partial(nonlinearity, kind=act)

        # compute in_ch_mult, block_in and curr_res at lowest res
        in_ch_mult = (1,) + tuple(ch_mult)
        block_in = ch * ch_mult[self.num_resolutions - 1]
        curr_res = resolution // 2 ** (self.num_resolutions - 1)
        self.z_shape = (1, z_channels, curr_res, curr_res)
        print(
            "Working with z of shape {} = {} dimensions.".format(
                self.z_shape, np.prod(self.z_shape)
            )
        )

        make_attn_cls = self._make_attn()
        make_resblock_cls = self._make_resblock()
        make_conv_cls = self._make_conv()
        # z to block_in
        self.conv_in = Conv2d(
            z_channels, block_in, kernel_size=3, stride=1, padding=1, circular=circular, coord=coord
        )

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = make_resblock_cls(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout,
            act=act,
            circular=circular,
            coord=coord
        )
        self.mid.attn_1 = make_attn_cls(block_in, attn_type=attn_type)
        self.mid.block_2 = make_resblock_cls(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout,
            act=act,
            circular=circular,
            coord=coord
        )

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks + 1):
                block.append(
                    make_resblock_cls(
                        in_channels=block_in,
                        out_channels=block_out,
                        temb_channels=self.temb_ch,
                        dropout=dropout,
                        act=act,
                        circular=circular,
                        coord=coord
                    )
                )
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(make_attn_cls(block_in, attn_type=attn_type))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv, resamp_single_side, circular=circular, coord=coord)
                curr_res = curr_res * 2
            self.up.insert(0, up)  # prepend to get consistent order

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = make_conv_cls(
            block_in, out_ch, kernel_size=3, stride=1, padding=1, circular=circular, coord=coord
        )

    def _make_attn(self) -> Callable:
        return make_attn

    def _make_resblock(self) -> Callable:
        return ResnetBlock

    def _make_conv(self) -> Callable:
        return Conv2d

    def get_last_layer(self, **kwargs):
        return self.conv_out.weight

    def forward(self, z, **kwargs):
        # assert z.shape[1:] == self.z_shape[1:]
        self.last_z_shape = z.shape

        # timestep embedding
        temb = None

        # z to block_in
        h = self.conv_in(z)

        # middle
        h = self.mid.block_1(h, temb, **kwargs)
        h = self.mid.attn_1(h, **kwargs)
        h = self.mid.block_2(h, temb, **kwargs)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h, temb, **kwargs)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h, **kwargs)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # end
        if self.give_pre_end:
            return h

        h = self.norm_out(h)
        h = self.nonlinearity(h)
        h = self.conv_out(h, **kwargs)
        if self.tanh_out:
            h = torch.tanh(h)
        return h

class SlicedConv(nn.Module):
    def __init__(
            self, 
            in_channels: int,
            out_channels: int,
            kernel_size=3,
            stride=1,
            padding=0,
            height=64,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.padding = padding
        self.in_channels = in_channels 
        assert stride == 1 or stride == 2
        self.out_channels = out_channels
        self.groups = height // 2 + padding
        self.height = height
        self.stride = stride
        self.conv = torch.nn.Conv1d(in_channels * (height + 2 * padding), 
                                    self.out_channels // stride * (height + 2 * padding), #  stride for downsample
                                    kernel_size, 
                                    stride, 
                                    padding=kernel_size//2 if stride==1 else 0,
                                    padding_mode='circular',
                                    groups=self.groups)
    
    def forward(self, x):
        assert x.shape[-1] == self.height
        x = torch.flatten(x.permute(0, 3, 1, 2), start_dim=1, end_dim=2) # (B, C, W, H) -> (B, H, C, W) -> (B, C*H, W)
        if self.padding:
            x = F.pad(x, (0, 0, self.in_channels, self.in_channels))
        if self.stride == 2:
            x = F.pad(x, (0, 1))
        x = self.conv(x)
        if self.padding:
            x = x[:, self.out_channels//self.stride:self.out_channels//self.stride*(1 + self.height), :]
        B, _, W = x.shape
        # (B, C_out*H_out, W_out) -> (B, H_out, C_out, W_out) -> (B, C_out, W_out, H_out)
        x = x.reshape(B, -1, self.out_channels, W).permute(0, 2, 3, 1) 
        return x

    @property
    def weight(self):
        return self.conv.weight

class SlicedUpsample(nn.Module):
    def __init__(self, in_channels, with_conv, height, padding=0):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = SlicedConv(in_channels, in_channels, kernel_size=3, stride=1, padding=padding, height=height*2)

    def forward(self, x):
        x = torch.nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x
    
class SlicedDownsample(nn.Module):
    def __init__(self, in_channels, with_conv, height, padding=0):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            # no asymmetric padding in torch conv, must do it ourselves
            self.conv = SlicedConv(in_channels, in_channels, kernel_size=3, stride=2, padding=padding, height=height)

    def forward(self, x):
        if self.with_conv:
            x = self.conv(x)
        else:
            x = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
        return x
    
class SlicedResnetBlock(nn.Module):
    def __init__(
        self,
        *,
        in_channels,
        out_channels=None,
        conv_shortcut=False,
        dropout,
        temb_channels=512,
        act = 'relu',
        padding=0,
        height=64, 
    ):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels)
        self.conv1 = SlicedConv(
            in_channels, out_channels, kernel_size=3, stride=1, padding=padding, height=height
        )
        if temb_channels > 0:
            self.temb_proj = torch.nn.Linear(temb_channels, out_channels)
        self.norm2 = Normalize(out_channels)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = SlicedConv(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1-padding, height=height
        )
        self.nonlinearity = partial(nonlinearity, kind=act)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = SlicedConv(
                    in_channels, out_channels, kernel_size=3, stride=1, padding=padding, height=height
                )
            else:
                self.nin_shortcut = SlicedConv(
                    in_channels, out_channels, kernel_size=1, stride=1, padding=padding, height=height
                )

    def forward(self, x, temb):
        h = x
        h = self.norm1(h)
        h = self.nonlinearity(h)
        h = self.conv1(h)

        if temb is not None:
            h = h + self.temb_proj(self.nonlinearity(temb))[:, :, None, None]

        h = self.norm2(h)
        h = self.nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x + h
    
## todo: sliced attention


class SlicedEncoder(nn.Module):
    def __init__(
        self,
        *,
        ch,
        out_ch,
        ch_mult=(1, 2, 4, 8),
        num_res_blocks,
        attn_resolutions,
        dropout=0.0,
        resamp_with_conv=True,
        resamp_single_side=False,
        in_channels,
        resolution=64, # for kitti360
        z_channels,
        double_z=True,
        use_linear_attn=False,
        attn_type="vanilla",
        edge_conv=False,
        azi=0,
        inc=0,
        act='relu',
        circular=False,
        **ignore_kwargs,
    ):
        super().__init__()
        if use_linear_attn:
            attn_type = "linear"
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.edge_conv = edge_conv
        self.nonlinearity = partial(nonlinearity, kind=act)

        # downsampling
        padding=0
        self.conv_in = SlicedConv(
            in_channels, self.ch, kernel_size=3, stride=1, padding=padding, height=resolution
        )
        padding=1-padding

        curr_res = resolution
        in_ch_mult = (1,) + tuple(ch_mult)
        self.in_ch_mult = in_ch_mult
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(
                    SlicedResnetBlock(
                        in_channels=block_in,
                        out_channels=block_out,
                        temb_channels=self.temb_ch,
                        dropout=dropout,
                        act=act,
                        padding=padding,
                        height=curr_res
                    )
                )
                if block_in != block_out:
                    padding = 1 - padding
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(make_attn(block_in, attn_type=attn_type))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:
                down.downsample = SlicedDownsample(
                                    in_channels=block_in, 
                                    with_conv=resamp_with_conv, 
                                    height=curr_res, 
                                    padding=padding)  
                if resamp_with_conv:
                    padding = 1 - padding
                curr_res = curr_res // 2
                azi *= 2
                inc *= 2
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = SlicedResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout,
            act=act,
            padding=padding,
            height=curr_res
        )
        padding = 1-padding
        self.mid.attn_1 = make_attn(block_in, attn_type=attn_type)
        self.mid.block_2 = SlicedResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout,
            act=act,
            padding=padding,
            height=curr_res
        ) 
        padding = 1-padding

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = SlicedConv(
            block_in,
            2 * z_channels if double_z else z_channels,
            kernel_size=3,
            stride=1,
            padding=padding,
            height=curr_res
        )

    def forward(self, x):
        # timestep embedding
        temb = None
        # breakpoint()
        # downsampling
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1], temb)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions - 1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        # middle
        h = hs[-1]
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # end
        h = self.norm_out(h)
        h = self.nonlinearity(h)
        h = self.conv_out(h)
        return h


class SlicedDecoder(nn.Module):
    def __init__(
        self,
        *,
        ch,
        out_ch,
        ch_mult=(1, 2, 4, 8),
        num_res_blocks,
        attn_resolutions,
        dropout=0.0,
        resamp_with_conv=True,
        resamp_single_side=False,
        in_channels,
        resolution,
        z_channels,
        give_pre_end=False,
        tanh_out=False,
        use_linear_attn=False,
        attn_type="vanilla",
        act='relu',
        circular=False,
        **ignorekwargs,
    ):
        super().__init__()
        if use_linear_attn:
            attn_type = "linear"
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.give_pre_end = give_pre_end
        self.tanh_out = tanh_out
        self.nonlinearity = partial(nonlinearity, kind=act)

        # compute in_ch_mult, block_in and curr_res at lowest res
        in_ch_mult = (1,) + tuple(ch_mult)
        block_in = ch * ch_mult[self.num_resolutions - 1]
        curr_res = resolution // 2 ** (self.num_resolutions - 1)
        self.z_shape = (1, z_channels, curr_res, curr_res)
        print(
            "Working with z of shape {} = {} dimensions.".format(
                self.z_shape, np.prod(self.z_shape)
            )
        )

        make_attn_cls = self._make_attn()
        make_resblock_cls = self._make_resblock()
        make_conv_cls = self._make_conv()
        # z to block_in
        padding=0
        self.conv_in = SlicedConv(
            z_channels, block_in, kernel_size=3, stride=1, padding=padding, height=curr_res
        )
        padding = 1-padding

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = make_resblock_cls(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout,
            act=act,
            padding=padding, 
            height=curr_res
        )
        padding = 1-padding
        self.mid.attn_1 = make_attn_cls(block_in, attn_type=attn_type)
        self.mid.block_2 = make_resblock_cls(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout,
            act=act,
            padding=padding, 
            height=curr_res
        )
        padding = 1-padding

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks + 1):
                block.append(
                    make_resblock_cls(
                        in_channels=block_in,
                        out_channels=block_out,
                        temb_channels=self.temb_ch,
                        dropout=dropout,
                        act=act,
                        padding=padding, 
                        height=curr_res
                    )
                )
                if block_in != block_out:
                    padding = 1-padding
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(make_attn_cls(block_in, attn_type=attn_type))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = SlicedUpsample(in_channels=block_in, 
                                            with_conv=resamp_with_conv, 
                                            height=curr_res,
                                            padding=padding)
                if resamp_with_conv:
                    padding = 1-padding
                curr_res = curr_res * 2
            self.up.insert(0, up)  # prepend to get consistent order

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = make_conv_cls(
            block_in, out_ch, kernel_size=3, stride=1, padding=padding, height=curr_res
        )

    def _make_attn(self) -> Callable:
        return make_attn

    def _make_resblock(self) -> Callable:
        return SlicedResnetBlock

    def _make_conv(self) -> Callable:
        return SlicedConv

    def get_last_layer(self, **kwargs):
        return self.conv_out.weight

    def forward(self, z, **kwargs):
        # assert z.shape[1:] == self.z_shape[1:]
        self.last_z_shape = z.shape

        # timestep embedding
        temb = None

        # z to block_in
        h = self.conv_in(z)

        # middle
        h = self.mid.block_1(h, temb, **kwargs)
        h = self.mid.attn_1(h, **kwargs)
        h = self.mid.block_2(h, temb, **kwargs)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h, temb, **kwargs)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h, **kwargs)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # end
        if self.give_pre_end:
            return h

        h = self.norm_out(h)
        h = self.nonlinearity(h)
        h = self.conv_out(h, **kwargs)
        if self.tanh_out:
            h = torch.tanh(h)
        return h
