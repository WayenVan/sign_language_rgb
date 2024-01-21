import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
from einops import rearrange
from ..modules.resnet import *
from ..modules.unet import *
from ..modules.tconv import *
from torch.cuda.amp.autocast_mode import autocast
from ..utils.decode import CTCDecoder
from ..utils.loss import GlobalLoss

__all__ = [
    'ResnetTransformer'
]

class SLRModel(nn.Module):
    def __init__(
        self, 
        backbone: nn.Module,
        vocab,
        loss_weight,
        loss_temp,
        ctc_search_type = 'beam',
        return_label=True,
        **kwargs
        ) -> None:
        super().__init__()
        
        self.vocab = vocab
        self.return_label = return_label
        
        self.backbone = backbone
        self.loss = GlobalLoss(loss_weight, loss_temp)
        
        self.decoder = CTCDecoder(self.vocab, blank_id=0, search_mode=ctc_search_type, log_probs_input=True)
    
    def forward(self, *args, **kwargs):
        backbone_out = self.backbone(*args, **kwargs)
        if self.return_label:
            y_predict = backbone_out['seq_out']
            video_length = backbone_out['video_length']
            y_predict = torch.nn.functional.log_softmax(y_predict, -1)
            backbone_out['seq_out_label'] = self.decoder(y_predict, video_length)
        return backbone_out
    
    def criterion(self, outputs, target, target_length): 
        return self.loss(outputs, target, target_length)
    
    def inference(self, *args, **kwargs) -> List[List[str]]:
        with torch.no_grad():
            outputs = self.backbone(*args, **kwargs)
            y_predict = outputs['seq_out']
            video_length = outputs['video_length']
            y_predict = torch.nn.functional.log_softmax(y_predict, -1)
        return self.decoder(y_predict, video_length)

class ResnetTransformer(nn.Module):
    
    def __init__(
        self,
        d_feedforward,
        n_head,
        n_class,
        resnet_type='resnet18'
        ) -> None:
        super().__init__()

        self.resnet: ResNet = resnet18(weights=ResNet18_Weights.DEFAULT)
        d_model = self.resnet.fc.in_features
        self.tconv = TemporalConv(d_model, 2*d_model)
        decoder_layer = nn.TransformerEncoderLayer(2*d_model, n_head, dim_feedforward=d_feedforward)
        self.trans_decoder = nn.TransformerEncoder(decoder_layer, 2)
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