from typing import Any, Tuple
import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange

class GlobalLoss:
    def __init__(self, weights, temperature) -> None:
        #weigts: ctc_seq, ctc_conv, distill
        self.CTC = nn.CTCLoss(blank=0, reduction='none')
        self.distll = SelfDistill(temperature)
        self.weights = weights
    
    def __call__(self, output, target, target_length) -> Any:
        #[t, n, c] logits
        conv_out = output['conv_out']
        seq_out = output['seq_out']
        input_length = output['video_length']
        
        conv_out, seq_out = F.log_softmax(conv_out, dim=-1), F.log_softmax(seq_out, dim=-1)

        loss = 0
        if self.weights[0] > 0.:
            loss += self.CTC(seq_out, target, input_length.cpu().int(), target_length.cpu().int()).mean()* self.weights[0]
        if self.weights[1] > 0.:
            loss += self.CTC(conv_out, target, input_length.cpu().int(), target_length.cpu().int()).mean() * self.weights[1]
        if self.weights[2] > 0.:
            loss += self.distll(seq_out, conv_out) * self.weights[2]

        return loss    
    
    def _filter_nan(self, *losses):
        ret = []
        for loss in losses:
            if torch.all(torch.isinf(loss)).item():
                loss: torch.Tensor
                print('loss is inf')
                loss = torch.nan_to_num(loss, posinf=0.)
            ret.append(loss)
        return tuple(ret)
            

        

class SelfDistill:

    def __init__(self, temperature) -> None:
        self.t = temperature
        
    def __call__(self, teacher, student) -> Any:
        # seq: logits [t, n, c]
        T, N, C = teacher.shape
        assert (T, N, C) == student.shape
        teacher, student = teacher/self.t, student/self.t

        teacher = F.log_softmax(rearrange(teacher, 't n c -> (t n) c'), dim=-1)
        student = F.log_softmax(rearrange(student, 't n c -> (t n) c'), dim=-1)
        return F.kl_div(student, teacher.detach(), log_target=True, reduction='batchmean')
