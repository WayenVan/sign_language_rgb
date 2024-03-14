

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Any
from einops import rearrange
from ..modules.resnet.resnet import *
from ..modules.tconv import *
from torch.cuda.amp.autocast_mode import autocast
from ..utils.decode import CTCDecoder
from ..modules.slr_base.base_stream import BaseStream

class SLRModel(nn.Module):
    def __init__(
        self, 
        backbone: BaseStream, #x3d & flownet
        vocab,
        ctc_search_type = 'greedy',
        return_label=True,
        **kwargs
        ) -> None:
        super().__init__()
        
        self.vocab = vocab
        self.return_label = return_label
        self.backbone = backbone
        self.loss = CTCLoss
        self.decoder = CTCDecoder(self.vocab, blank_id=0, search_mode=ctc_search_type, log_probs_input=True)
    
    def forward(self, input, t_length, *args, **kwargs):
        backbone_out = self.backbone(input, t_length)

        if self.return_label:
            y_predict = backbone_out['out']
            video_length = backbone_out['t_length']
            y_predict = torch.nn.functional.log_softmax(y_predict, -1).detach().cpu()
            backbone_out['out_labels'] = self.decoder(y_predict, video_length)
        return backbone_out
    
    def criterion(self, outputs, target, target_length): 
        return self.loss(outputs['encoder_out']['out'], outputs['out'], outputs['t_length'], target, target_length)

    @torch.no_grad()
    def inference(self, *args, **kwargs) -> List[List[str]]:
        outputs = self.backbone(*args, **kwargs)
        y_predict = outputs['out']
        video_length = outputs['t_length']
        y_predict = torch.nn.functional.log_softmax(y_predict, -1).detach().cpu()
        return self.decoder(y_predict, video_length)
