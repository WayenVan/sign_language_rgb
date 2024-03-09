import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
from einops import rearrange
from ...modules.resnet.resnet import *
from ...modules.tconv import *
from ...modules.tadaresnet import TadaResnet

class TadaConvTransformer(nn.Module):
    
    def __init__(
        self,
        n_head,
        n_class,
        n_layers,
        d_feedforward=1024
        ) -> None:
        super().__init__()

        self.tadaresnet = TadaResnet()
        d_model = 512
        self.tconv = TemporalConv(d_model, 2*d_model)

        self.ln = nn.LayerNorm(2*d_model)
        encoder_layer = nn.TransformerEncoderLayer(2*d_model, n_head, dim_feedforward=d_feedforward)
        self.trans_decoder = nn.TransformerEncoder(encoder_layer, n_layers)

        self.fc_conv = nn.Linear(2*d_model, n_class)
        self.fc = nn.Linear(2*d_model, n_class)
    
    
    def forward(self, x, video_length):
        """
        :param x: [t, n, c, h, w]
        :param video_length: [n]
        """
        batch_size = x.size(dim=1)
        x = self.tadaresnet(x)
        #[t n c]

        x = rearrange(x, 't n c -> n c t', n=batch_size)
        x, video_length = self.tconv(x, video_length)
        x = rearrange(x, 'n c t -> t n c')
        conv_out = self.fc_conv(x)
        
        mask = self._make_video_mask(video_length, x.size(dim=0))
        x = self.ln(x)
        x = self.trans_decoder(x, src_key_padding_mask=mask)
        x = self.fc(x)

        return dict(
            seq_out=x,
            conv_out=conv_out,
            video_length=video_length 
        )
    
    @staticmethod
    def _make_video_mask(video_length: torch.Tensor, temporal_dim):
        batch_size = video_length.size(dim=0)
        mask = torch.ones(batch_size, temporal_dim)
        for idx in range(batch_size):
            mask[idx, :video_length[idx]] = 0
        return mask.bool().to(video_length.device)