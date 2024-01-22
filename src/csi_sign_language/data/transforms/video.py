import math
from typing import Any
import numpy as np
from ...utils.data import *
from einops import rearrange
from torchvision import transforms as T
import torch


class Standization:

    def __init__(self, mean, std, epsilon=1e-5) -> None:
        """
        :param mean: size 3, channel means
        :param var: size 3, channel stds
        :param epsilon: -5
        """
        self.mean = torch.tensor(mean, dtype=torch.float32)
        self.std = torch.tensor(std, dtype=torch.float32)
        self.epsilon = epsilon

    def __call__(self, data) -> Any:
        video = data['video']
        #t, c, h, wepsilon
        video = (
            (video - self._rearrange(self.mean)) /
            torch.sqrt(self._rearrange(self.std)**2 + self.epsilon)
        )
        data['video'] = video
        return data
        
    def _rearrange(self, x):
        return rearrange(x, '(t c h w) -> t c h w', t=1, h=1, w=1)
            

class DownSampleT:
    def __init__(self, step) -> None:
        self.s = step
        
    def __call__(self, data) -> Any:
        data['video'] = data['video'][::self.s]
        return data
        
class FrameScale:
    def __init__(self, min, max) -> None:
        self.min = min
        self.max = max

    def __call__(self, data) -> Any:
        video = data['video']
        data['video'] = self.min + video * (self.max - self.min)
        return data

class ToTensor:
    def __init__(self, keys) -> None:
        self.keys = keys
    
    def __call__(self, data) -> Any:
        for k, v in data.items():
            if k in self.keys:
                data[k] = torch.tensor(v)
        return data
    

class CentralCrop:

    def __init__(self, size=224) -> None:
        self.size = size

    def __call__(self, data) -> Any:
        video: torch.Tensor = data['video']
        T, C, H, W = video.shape
        start_h = math.floor((H - self.size)/2.)
        start_w = math.floor((W - self.size)/2.)
        data['video'] = video[:, :, start_h:start_h+self.size, start_w:start_w+self.size]
        return data
        