from typing import Any
import numpy as np
from ...utils.data import *
from einops import rearrange

class ImageResize():
    
    def __init__(self, h, w) -> None:
        self.h = h
        self.w = w
        
    def __call__(self, data) -> Any:
        """
        :param data: [h w c] the image to change
        """
        return cv2.resize(data, (self.h, self.w))
        
