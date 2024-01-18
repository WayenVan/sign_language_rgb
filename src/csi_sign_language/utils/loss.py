from typing import Any
import torch 
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class Losss:
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
        return self.weights[0]*self.CTC_seq(seq_out, target, input_length, target_length) + \
            self.weights[1]*self.CTC_conv(conv_out, target, input_length, target_length) + \
            self.weights[2]*self.distll(seq_out, conv_out)


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
        return F.kl_div(student, teacher, log_target=True, reduction='batchmean')
