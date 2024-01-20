from typing import Any
import numpy as np
from ...utils.data import *
from einops import rearrange
from torchvision import transforms as T
import torch

# class Compose:
#     def __init__(self, transforms) -> None:
#         self.transforms = transforms
    
#     def __call__(self, video, gloss) -> Any:
#         for t in self.transforms:
#             video, gloss = t(video, gloss)
#         return video, gloss

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
    def __init__(self) -> None:
        pass
    def __call__(self, data) -> Any:
        data['video'] = data['video'].astype(np.float32) / 255.
        return data

class ToTensor:
    def __init__(self, keys) -> None:
        self.keys = keys
    
    def __call__(self, data) -> Any:
        for k, v in data.items():
            if k in self.keys:
                data[k] = torch.tensor(v)
        return data
    
