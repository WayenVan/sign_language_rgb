import math
from typing import Any
import numpy as np
from ...utils.data import *
from einops import rearrange
from torchvision import transforms as T
import torch
import numbers
import random
import copy

class Standization:

    def __init__(self, mean, std, epsilon=1e-5) -> None:
        """
        input should be a tensor
        :param mean: size 3, channel means
        :param var: size 3, channel stds
        :param epsilon: -5
        """
        self.mean = np.array(mean, dtype='float32')
        self.std = np.array(std, dtype='float32')
        self.epsilon = epsilon

    def __call__(self, data) -> Any:
        video = data['video']
        #t, c, h, wepsilon
        video = (
            (video - self._rearrange(self.mean)) /
            np.sqrt(self._rearrange(self.std)**2 + self.epsilon)
        )
        data['video'] = video
        return data
        
    def _rearrange(self, x):
        return rearrange(x, '(t c h w) -> t c h w', t=1, h=1, w=1)
            

class TemporalDownSampleT:
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

class RandomCrop(object):
    """
    Extract random crop of the video.
    Args:
        size (sequence or int): Desired output size for the crop in format (h, w).
        crop_position (str): Selected corner (or center) position from the
        list ['c', 'tl', 'tr', 'bl', 'br']. If it is non, crop position is
        selected randomly at each call.
    """

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            if size < 0:
                raise ValueError('If size is a single number, it must be positive')
            size = (size, size)
        else:
            if len(size) != 2:
                raise ValueError('If size is a sequence, it must be of len 2.')
        self.size = size

    def __call__(self, data):
        clip = data['video']

        crop_h, crop_w = self.size
        im_c, im_h, im_w = clip[0].shape

        if crop_w > im_w:
            pad = crop_w - im_w
            clip = [np.pad(img, ((0, 0), (0, 0), (pad // 2, pad - pad // 2)), 'constant', constant_values=0) for img in
                    clip]
            w1 = 0
        else:
            w1 = random.randint(0, im_w - crop_w)

        if crop_h > im_h:
            pad = crop_h - im_h
            clip = [np.pad(img, ((0, 0), (pad // 2, pad - pad // 2), (0, 0)), 'constant', constant_values=0) for img in
                    clip]
            h1 = 0
        else:
            h1 = random.randint(0, im_h - crop_h)

        data['video'] = np.stack([img[:, h1:h1 + crop_h, w1:w1 + crop_w] for img in clip])
        return data
class RandomHorizontalFlip(object):

    def __init__(self, prob):
        self.prob = prob

    def __call__(self, data):
        clip = data['video']
        # B, H, W, 3
        flag = random.random() < self.prob
        if flag:
            clip = np.flip(clip, axis=-1)
            clip = np.ascontiguousarray(copy.deepcopy(clip))
        data['video'] = np.array(clip)
        return data

class TemporalRescale(object):

    def __init__(self, temp_scaling=0.2, min_len=32, max_len=230):
        self.min_len = min_len
        self.max_len = max_len
        self.L = 1.0 - temp_scaling
        self.U = 1.0 + temp_scaling

    def __call__(self, data):
        clip = data['video']
        vid_len = len(clip)
        new_len = int(vid_len * (self.L + (self.U - self.L) * np.random.random()))
        if new_len < self.min_len:
            new_len = self.min_len
        if new_len > self.max_len:
            new_len = self.max_len
        if (new_len - 4) % 4 != 0:
            new_len += 4 - (new_len - 4) % 4
        if new_len <= vid_len:
            index = sorted(random.sample(range(vid_len), new_len))
        else:
            index = sorted(random.choices(range(vid_len), k=new_len))
        data['video'] = clip[index]
        return data