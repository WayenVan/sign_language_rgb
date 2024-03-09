import torch
from torch import nn
from ..modules.slr_streams.base_stream import BaseStream

class SLRTwoStream(nn.Module):
    
    def __init__(
        self,
        v_stream: BaseStream,
        f_stream: BaseStream,
        *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.ctc_loss = nn.CTCLoss(blank=0)
        self.log_soft_max = nn.LogSoftmax(dim=-1)

        self.v_stream = v_stream
        self.f_stream = f_stream
    
    def forward(self, x, t_length):
        v_out = self.v_stream(x, t_length)
        f_out = self.f_stream(x, t_length)
        
        return dict(
            v_out=v_out,
            f_out=f_out
        )

    def citerion(self, outputs, target, target_length):

        v_out = outputs['v_out']
        f_out = outputs['f_out']

        v_loss = self.ctc_loss(
            self.log_soft_max(v_out['out']),
            target,
            v_out['t_length'].cpu().int(),
            target_length.cpu().int()
        ).mean()

        f_loss = self.ctc_loss(
            self.log_soft_max(f_out['out']),
            target,
            f_out['t_length'].cpu().int(),
            target_length.cpu().int()
        ).mean()
        
        return v_loss + f_loss
        