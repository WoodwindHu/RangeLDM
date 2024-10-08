from contextlib import nullcontext
from functools import partial
from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn

class AbstractEmbModel(nn.Module):
    def __init__(self):
        super().__init__()
        self._is_trainable = None
        self._ucg_rate = None
        self._input_key = None
        self.dtype = torch.float32

    @property
    def is_trainable(self) -> bool:
        return self._is_trainable

    @property
    def ucg_rate(self) -> Union[float, torch.Tensor]:
        return self._ucg_rate

    @property
    def input_key(self) -> str:
        return self._input_key

    @is_trainable.setter
    def is_trainable(self, value: bool):
        self._is_trainable = value

    @ucg_rate.setter
    def ucg_rate(self, value: Union[float, torch.Tensor]):
        self._ucg_rate = value

    @input_key.setter
    def input_key(self, value: str):
        self._input_key = value

    @is_trainable.deleter
    def is_trainable(self):
        del self._is_trainable

    @ucg_rate.deleter
    def ucg_rate(self):
        del self._ucg_rate

    @input_key.deleter
    def input_key(self):
        del self._input_key

class IdentityEncoder(AbstractEmbModel):
    def encode(self, x):
        return x

    def forward(self, x):
        return x
    
class SparseRangeImageEncoder(AbstractEmbModel):
    def __init__(self, outdim=4, middle=32, densification=4):
        super().__init__()
        # downsample along W
        self.conv1 = torch.nn.Conv2d(
                2, middle, kernel_size=3, stride=(2, 1), 
                padding=0
            )
        self.act = torch.nn.SiLU()
        self.conv2 = torch.nn.Conv2d(
                middle, outdim, kernel_size=3, stride=(2, 1), 
                padding=0
            )
        self.densification = densification

    def encode(self, x):
        return self(x)

    def forward(self, x):
        x = torch.nn.functional.pad(x, (0, 0, 0, 1), mode="circular")
        x = torch.nn.functional.pad(x, (1, 1, 0, 0), mode="constant")
        x = self.conv1(x)
        x = self.act(x)
        x = torch.nn.functional.pad(x, (0, 0, 0, 1), mode="circular")
        x = torch.nn.functional.pad(x, (1, 1, 0, 0), mode="constant")
        x = self.conv2(x)
        return x
    
class SparseRangeImageEncoder2(AbstractEmbModel):
    def encode(self, x):
        return self(x)

    def forward(self, x):
        # (B, C, W, H) -> (B, C*4, W/4, H), nearest 4 pixels to 1 pixel
        B, C, W, H = x.shape
        x = torch.flatten(x.permute(0, 2, 1, 3), start_dim=1, end_dim=2) # (B, C, W, H) -> (B, W, C, H) -> (B, W*C, H)
        x = x.reshape(B, W//4, C*4, H).permute(0, 2, 1, 3) # (B, W*C, H) -> (B, W//4, C*4, H) -> (B, C*4, W//4, H)
        return x