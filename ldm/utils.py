import diffusers 
import torch 
from typing import Any, Callable, Optional, Union

from torch.nn.common_types import _size_2_t
import torch.nn.functional as F
from torch.nn.modules.utils import _pair



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

    def forward(self, input, scale=1.0):
        return self._conv_forward(input, self.weight, self.bias)
    
class Downsample2D(torch.nn.Module):
    """A 2D downsampling layer with an optional convolution.

    Parameters:
        channels (`int`):
            number of channels in the inputs and outputs.
        use_conv (`bool`, default `False`):
            option to use a convolution.
        out_channels (`int`, optional):
            number of output channels. Defaults to `channels`.
        padding (`int`, default `1`):
            padding for the convolution.
        name (`str`, default `conv`):
            name of the downsampling 2D layer.
    """

    def __init__(
        self,
        channels: int,
        use_conv: bool = False,
        out_channels: Optional[int] = None,
        padding: int = 1,
        name: str = "conv",
    ):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.padding = padding
        stride = 2
        self.name = name

        if use_conv:
            conv = Conv2d(self.channels, self.out_channels, 3, stride=stride, padding=padding, circular=True)
        else:
            assert self.channels == self.out_channels
            conv = torch.nn.AvgPool2d(kernel_size=stride, stride=stride)

        # TODO(Suraj, Patrick) - clean up after weight dicts are correctly renamed
        if name == "conv":
            self.Conv2d_0 = conv
            self.conv = conv
        elif name == "Conv2d_0":
            self.conv = conv
        else:
            self.conv = conv

    def forward(self, hidden_states, scale: float = 1.0):
        assert hidden_states.shape[1] == self.channels
        if self.use_conv and self.padding == 0:
            hidden_states = torch.nn.functional.pad(hidden_states, (0, 0, 0, 1), mode="circular")
            hidden_states = torch.nn.functional.pad(hidden_states, (0, 1, 0, 0), mode="constant")

        assert hidden_states.shape[1] == self.channels
        hidden_states = self.conv(hidden_states)

        return hidden_states
    
class attn_identity(torch.nn.Identity):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, input, **kwargs):
        return input

def replace_conv(module, name='Conv2d'):
        '''
        Recursively put desired conv2d in nn.module module.

        set module = net to start code.
        '''
        # go through all attributes of module nn.module (e.g. network or layer) and put batch norms if present
        for attr_str in dir(module):
            target_attr = getattr(module, attr_str)
            if type(target_attr) == diffusers.models.lora.LoRACompatibleConv or type(target_attr) == torch.nn.Conv2d:
                # print('replaced: ', name, attr_str)
                new_bn = Conv2d(target_attr.in_channels, 
                                target_attr.out_channels,
                                kernel_size = target_attr.kernel_size, 
                                stride=target_attr.stride, 
                                padding=target_attr.padding,
                                circular=True)
                setattr(module, attr_str, new_bn)

        # iterate through immediate child modules. Note, the recursion is done by our code no need to use named_modules()
        for name, immediate_child_module in module.named_children():
            replace_conv(immediate_child_module, name)

def replace_attn(module, name='Attention'):
    '''
    Recursively put desired Attention in nn.module module.

    set module = net to start code.
    '''
    # go through all attributes of module nn.module (e.g. network or layer) and put batch norms if present
    for attr_str in dir(module):
        target_attr = getattr(module, attr_str)
        if type(target_attr) == diffusers.models.attention_processor.Attention:
            # print('replaced: ', name, attr_str)
            new_bn = attn_identity()
            setattr(module, attr_str, new_bn)
        elif type(target_attr) == torch.nn.ModuleList:
            for i, attn in enumerate(target_attr):
                if type(attn) == diffusers.models.attention_processor.Attention:
                    # print('replaced: ', name, attr_str)
                    new_bn = attn_identity()
                    target_attr[i] = new_bn

    # iterate through immediate child modules. Note, the recursion is done by our code no need to use named_modules()
    for name, immediate_child_module in module.named_children():
        replace_attn(immediate_child_module, name)


def replace_down(module, name='down'):
        '''
        Recursively put desired conv2d in nn.module module.

        set module = net to start code.
        '''
        # go through all attributes of module nn.module (e.g. network or layer) and put batch norms if present
        for attr_str in dir(module):
            target_attr = getattr(module, attr_str)
            if type(target_attr) == diffusers.models.resnet.Downsample2D:
                # print('replaced: ', name, attr_str)
                new_bn = Downsample2D(target_attr.channels,
                                      use_conv=target_attr.use_conv,
                                      out_channels=target_attr.out_channels,
                                      padding=target_attr.padding,
                                      name=target_attr.name)
                setattr(module, attr_str, new_bn)
            elif type(target_attr) == torch.nn.ModuleList:
                for i, attn in enumerate(target_attr):
                    if type(attn) == diffusers.models.resnet.Downsample2D:
                        # print('replaced: ', name, attr_str)
                        new_bn = Downsample2D(attn.channels,
                                      use_conv=attn.use_conv,
                                      out_channels=attn.out_channels,
                                      padding=attn.padding,
                                      name=attn.name)
                        target_attr[i] = new_bn

        # iterate through immediate child modules. Note, the recursion is done by our code no need to use named_modules()
        for name, immediate_child_module in module.named_children():
            replace_down(immediate_child_module, name)