from typing import Any, Tuple
import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange

class GlobalLoss:
    def __init__(self, weights, temperature) -> None:
        #weigts: ctc_seq, ctc_conv, distill
        self.CTC_seq = nn.CTCLoss()
        self.CTC_conv = nn.CTCLoss()
        self.distll = SelfDistill(temperature)
        self.weights = weights
    
    def __call__(self, output, target, target_length) -> Any:
        #[t, n, c] logits
        conv_out = output['conv_out']
        seq_out = output['seq_out']
        input_length = output['video_length']

        conv_out, seq_out = F.log_softmax(conv_out, dim=-1), F.log_softmax(seq_out, dim=-1)
        seq_loss = self.CTC_seq(seq_out, target, input_length, target_length)
        conv_loss = self.CTC_conv(conv_out, target, input_length, target_length)
        distll_loss = self.distll(seq_out, conv_out)

        seq_loss, conv_loss, distll_loss = self._filter_nan((seq_loss, conv_loss, distll_loss))
        
        return self.weights[0]*seq_loss + \
            self.weights[1]*conv_loss + \
            self.weights[2]*distll_loss
    
    def _filter_nan(self, losses: Tuple):
        ret = []
        for idx, loss in enumerate(losses):
            if np.isnan(loss.item()) or np.isinf(loss.item()):
                print(f'warning, loss is nan or inf, index {idx}, value {loss.item()}')
                ret.append(0)
            else:
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
