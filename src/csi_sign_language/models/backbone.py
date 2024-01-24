import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
from einops import rearrange
from ..modules.resnet import *
from ..modules.unet import *
from ..modules.tconv import *
from ..modules.bilstm import BiLSTMLayer
from torch.cuda.amp.autocast_mode import autocast
from ..utils.decode import CTCDecoder
from ..modules.loss import GlobalLoss

class ResnetTransformer(nn.Module):
    
    def __init__(
        self,
        d_feedforward,
        n_head,
        n_class,
        layers,
        resnet_type='resnet18'
        ) -> None:
        super().__init__()

        self.resnet: ResNet = resnet18(weights=ResNet18_Weights.DEFAULT)
        d_model = self.resnet.fc.in_features
        self.tconv = TemporalConv(d_model, 2*d_model)
        encoder_layer = nn.TransformerEncoderLayer(2*d_model, n_head, dim_feedforward=d_feedforward)
        self.trans_decoder = nn.TransformerEncoder(encoder_layer, layers)
        self.fc_conv = nn.Linear(2*d_model, n_class)
        self.fc = nn.Linear(2*d_model, n_class)
    
    
    def forward(self, x, video_length):
        """
        :param x: [t, n, c, h, w]
        :param video_length: [n]
        """
        batch_size = x.size(dim=1)
        x = rearrange(x, 't n c h w -> (t n) c h w')
        x = self.resnet(x)
        x = rearrange(x, '(t n) c -> n c t', n=batch_size)

        x, video_length = self.tconv(x, video_length)
        x = rearrange(x, 'n c t -> t n c')
        conv_out = self.fc_conv(x)
        
        mask = self._make_video_mask(video_length, x.size(dim=0))
        x = x + self.trans_decoder(x, src_key_padding_mask=mask)
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


class ResnetLSTM(nn.Module):
    
    def __init__(
        self,
        n_class,
        n_layers,
        resnet_type='resnet18'
        ) -> None:
        super().__init__()

        self.resnet: ResNet = resnet18(weights=ResNet18_Weights.DEFAULT)
        d_model = self.resnet.fc.in_features
        self.tconv = TemporalConv(d_model, 2*d_model)
        self.rnn = BiLSTMLayer(2*d_model, hidden_size=2*d_model, num_layers=n_layers, bidirectional=True)
        self.fc_conv = nn.Linear(2*d_model, n_class)
        self.fc = nn.Linear(2*d_model, n_class)
    
    
    def forward(self, x, video_length):
        """
        :param x: [t, n, c, h, w]
        :param video_length: [n]
        """
        batch_size = x.size(dim=1)
        x = rearrange(x, 't n c h w -> (t n) c h w')
        x = self.resnet(x)
        x = rearrange(x, '(t n) c -> n c t', n=batch_size)

        x, video_length = self.tconv(x, video_length)
        x = rearrange(x, 'n c t -> t n c')
        conv_out = self.fc_conv(x)
        
        x = self.rnn(x, video_length)['predictions']
        x = self.fc(x)

        return dict(
            seq_out=x,
            conv_out=conv_out,
            video_length=video_length 
        )