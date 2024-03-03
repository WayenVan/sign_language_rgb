
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
        x3d: nn.Module,
        neck_projection: nn.Module,
        sequence_model: nn.Module,
        header: nn.Module,
        spatial_aux_header: nn.Module,
        temporal_aux_header: nn.Module,
        vocab,
        ctc_search_type = 'greedy',
        sequence_copy_weights = True,
        **kwargs
        ) -> None:
        super().__init__()
        
        self.x3d = x3d
        self.neck_proj = neck_projection
        self.seq_model = sequence_model
        self.spatial_aux = spatial_aux_header
        self.temporal_aux = temporal_aux_header
        self.header = header
        
        self.vocab = vocab 
        self.copy_weights = sequence_copy_weights
        self.ctc_loss = nn.CTCLoss(blank=0)
        self.decoder =CTCDecoder(self.vocab, blank_id=0, search_mode=ctc_search_type, log_probs_input=True)

        if self.copy_weights:
            self._copy_weights_setup()
    
    def _copy_weights_setup(self):
        def hook(seq_module, inputs):
            if self.training:
                #move parameters inplace
                seq_module.load_state_dict(self.seq_model.state_dict())
        
        for module in [self.spatial_aux, self.temporal_aux]:
            seq_module = module.seqeunce_model

            #freeze parameters of sequence model
            for p in seq_module.parameters():
                p.requires_grad = False

            #regist hook
            seq_module.register_forward_pre_hook(hook)
    
    def forward(self, video, video_length):
        _, stages = self.x3d(video)
        
        x, video_length = self.neck_proj(stages[-1], video_length)
        # n c t
        x = rearrange(x, 'n c t -> t n c')
        x = self.seq_model(x)
        out = self.header(x)
        temporal_aux_out = None
        spatial_aux_out = None
        return dict(
            out = out,
            video_length = video_length,
            spatial_aux_out = spatial_aux_out,
            temporal_aux_out = temporal_aux_out
        )
    
    def criterion(self, outputs, target, target_length): 
        """
        :param target: [n, u]
        :param target_length: [n]
        :param outputss:
            out [t n c]
            spatial_out [t n d], [n]
            temporal_aux_out [t h w n d], [n h w]
        """
        loss = 0.
        
        #main ctc loss
        self.ctc_loss()
        
        #spatial ctc loss
        
        #temporal ctc loss
        
        return 

    @torch.no_grad()
    def inference(self, *args, **kwargs) -> List[List[str]]:
        outputs = self.backbone(*args, **kwargs)
        y_predict = outputs['seq_out']
        video_length = outputs['video_length']
        y_predict = torch.nn.functional.log_softmax(y_predict, -1).detach().cpu()
        return self.decoder(y_predict, video_length)
