import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Any
from einops import rearrange
from ..utils.decode import CTCDecoder
from ..modules.slr_base.base_stream import BaseStream
from collections import namedtuple


class SLRModel(nn.Module):
    def __init__(
        self, 
        backbone: BaseStream,
        vocab,
        loss: nn.Module = None,
        ctc_search_type = 'greedy',
        return_label=True,
        **kwargs
        ) -> None:
        super().__init__()
        
        self.loss = loss
        self.vocab = vocab
        self.return_label = return_label
        self.backbone = backbone
        self.decoder = CTCDecoder(self.vocab, blank_id=0, search_mode=ctc_search_type, log_probs_input=True)
    
    def forward(self, input, t_length, *args, **kwargs):
        #define return tuple
        SLRModelOut = namedtuple('SLRModelOut', ['backbone_out', 'label'])

        backbone_out = self.backbone(input, t_length)
        if self.return_label:
            y_predict = backbone_out.out
            video_length = backbone_out.t_length
            y_predict = torch.nn.functional.log_softmax(y_predict, -1).detach().cpu()
            label = self.decoder(y_predict, video_length)
            return SLRModelOut(backbone_out, label)
        return SLRModelOut(backbone_out, None)
    
    def criterion(self, model_out, input, input_length, target, target_length): 
        if self.loss is None:
            raise NotImplementedError()
        return self.loss(model_out.backbone_out, input, input_length, target, target_length)

    @torch.no_grad()
    def inference(self, *args, **kwargs) -> List[List[str]]:
        outputs = self.backbone(*args, **kwargs)
        y_predict = outputs.out
        video_length = outputs.t_length
        y_predict = torch.nn.functional.log_softmax(y_predict, -1).detach().cpu()
        return self.decoder(y_predict, video_length)
