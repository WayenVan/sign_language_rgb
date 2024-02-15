import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
from einops import rearrange
from ..modules.resnet.resnet import *
from ..modules.tconv import *
from torch.cuda.amp.autocast_mode import autocast
from ..utils.decode import CTCDecoder
from ..modules.loss import GlobalLoss

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
            y_predict = torch.nn.functional.log_softmax(y_predict, -1).detach().cpu()
            backbone_out['seq_out_label'] = self.decoder(y_predict, video_length)
        return backbone_out
    
    def criterion(self, outputs, target, target_length): 
        return self.loss(outputs, target, target_length)
    
    def inference(self, *args, **kwargs) -> List[List[str]]:
        with torch.no_grad():
            outputs = self.backbone(*args, **kwargs)
            y_predict = outputs['seq_out']
            video_length = outputs['video_length']
            y_predict = torch.nn.functional.log_softmax(y_predict, -1).detach().cpu()
        return self.decoder(y_predict, video_length)
