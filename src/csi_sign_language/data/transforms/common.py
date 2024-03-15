
import random
import math
import torch
import numpy as np
from einops import rearrange

class ApplyByKey():
    def __init__(self, key, transforms: list) -> None:
        self.key = key
        self.transforms = transforms
    
    def __call__(self, data, *args: torch.Any, **kwds: torch.Any) -> torch.Any:
        d = data[self.key]
        for t in self.transforms:
            d = t(d)
        data[self.key] = d
        return data

class Rearrange():
    def __init__(self, pattern: str) -> None:
        self.pattern = pattern
    
    def __call__(self, data) -> torch.Any:
        return rearrange(data, self.pattern)

class ToTensor:
    def __init__(self, dtype='default') -> None:
        self.dtype = dtype 
        self.str2dtype = {
            'float32': torch.float32,
            'float64': torch.double,
            'default': None
        }
    
    def __call__(self, data):
        data = torch.tensor(data, dtype=self.str2dtype[self.dtype])
        return data

class Rescale:
    def __init__(self, input, output) -> None:
        self.input = input
        self.output = output
    def __call__(self, video):
        video = self.output[0] + (self.output[1] - self.output[0]) * (video - self.input[0]) / (self.input[1] - self.input[0])
        return video

class CentralCrop:
    def __init__(self, size=224) -> None:
        self.size = size

    def __call__(self, video):
        T, C, H, W = video.shape
        start_h = math.floor((H - self.size)/2.)
        start_w = math.floor((W - self.size)/2.)
        video = video[:, :, start_h:start_h+self.size, start_w:start_w+self.size]
        return video
