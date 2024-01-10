import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from ..modules.resnet import *
from ..modules.unet import *

__all__ = [
    'ResnetTransformer'
]
    

class ResnetTransformer(nn.Module):
    
    def __init__(
        self,
        d_feedforward,
        n_head,
        n_class,
        resnet_type='resnet18') -> None:
        super().__init__()

        self.resnet: ResNet = resnet18(weights=ResNet18_Weights.DEFAULT)
        
        d_model = self.resnet.fc.in_features

        self.trans_decoder = nn.TransformerEncoderLayer(d_model, n_head, dim_feedforward=d_feedforward)
        self.fc = nn.Linear(d_model, n_class)

    def forward(self, x, frame_mask=None):
        """
        :param x: [t, n, c, h, w]
        """
        batch_size = x.size(dim=1)
        x = rearrange(x, 't n c h w -> (t n) c h w')
        x = self.resnet(x)
        x = rearrange(x, '(t n) d -> t n d', n=batch_size)
        x = self.trans_decoder(x, src_key_padding_mask=frame_mask)
        x = self.fc(x)
        x = F.log_softmax(x, dim=-1)

        return x
        