
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
    def __init__(self, keys, dtypes) -> None:
        self.keys = keys
        self.dtypes = dict(zip(keys, dtypes))
        
        self.str2dtype = {
            'float32': torch.float32,
            'float64': torch.double,
            'default': None
        }
    
    def __call__(self, data):
        for k, v in data.items():
            if k in self.keys:
                data[k] = torch.tensor(v, dtype=self.str2dtype[self.dtypes[k]])
        return data

class FrameScale:
    def __init__(self, min, max, input_range, key='video') -> None:
        self.min = min
        self.max = max
        self.input_range = input_range

    def __call__(self, video):
        video.astype('float32')
        video = self.min + (self.max - self.min) * (video - self.input_range[0]) / (self.input_range[1] - self.input_range[0])
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
