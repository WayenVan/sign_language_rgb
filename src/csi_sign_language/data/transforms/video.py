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
    def __init__(self) -> None:
        pass
    
    def __call__(self, data) -> Any:
        ret = {}
        for k, v in data.items():
            if k != '__key__':
                ret[k] = torch.tensor(v)
        return ret
    
