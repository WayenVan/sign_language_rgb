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

    def forward(self, x, video_length=None):
        """
        :param x: [t, n, c, h, w]
        :param video_length: [n]
        """
        batch_size = x.size(dim=1)
        temporal_dim = x.size(dim=0)
        x = rearrange(x, 't n c h w -> (t n) c h w')
        x = self.resnet(x)
        x = rearrange(x, '(t n) d -> t n d', n=batch_size)
        mask = self._make_video_mask(video_length, temporal_dim)
        x = self.trans_decoder(x, src_key_padding_mask=mask)
        x = self.fc(x)
        x = F.log_softmax(x, dim=-1)

        return x
    
    @staticmethod
    def _make_video_mask(video_length: torch.Tensor, temporal_dim):
        batch_size = video_length.size(dim=0)
        mask = torch.ones(batch_size, temporal_dim)
        for idx in range(batch_size):
            mask[idx, :video_length[idx]] = 0
        return mask.bool().to(video_length.device)