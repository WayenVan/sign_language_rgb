import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Any
from einops import rearrange
from ..utils.decode import CTCDecoder
from ..modules.slr_base.base_stream import BaseStream
from collections import namedtuple

class GlobalLoss:

    def __init__(self, weights, temperature) -> None:
        #weigts: ctc_seq, ctc_conv, distill
        self.CTC = nn.CTCLoss(blank=0, reduction='none')
        self.distll = SelfDistill(temperature)
        self.weights = weights
    
    def __call__(self, conv_out, seq_out, length, target, target_length) -> Any:
        #[t, n, c] logits
        input_length = length
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

class SLRModel(nn.Module):
    def __init__(
        self, 
        backbone: BaseStream,
        vocab,
        loss_weight,
        loss_temp,
        ctc_search_type = 'greedy',
        return_label=True,
        **kwargs
        ) -> None:
        super().__init__()
        
        self.vocab = vocab
        self.return_label = return_label
        self.backbone = backbone
        self.loss = GlobalLoss(loss_weight, loss_temp)
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
    
    def criterion(self, outputs, target, target_length): 
        encoder_out = outputs.backbone_out.encoder_out.out
        seq_out = outputs.backbone_out.out
        t_length = outputs.backbone_out.t_length
        return self.loss(encoder_out, seq_out, t_length, target, target_length)

    @torch.no_grad()
    def inference(self, *args, **kwargs) -> List[List[str]]:
        outputs = self.backbone(*args, **kwargs)
        y_predict = outputs.out
        video_length = outputs.t_length
        y_predict = torch.nn.functional.log_softmax(y_predict, -1).detach().cpu()
        return self.decoder(y_predict, video_length)
