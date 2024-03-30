import torch
from torch import nn
from mmengine.config import Config
from mmpose.apis import init_model
from csi_sign_language.utils.data import mapping_0_1
from einops import rearrange
import torch.nn.functional as F


class VACLoss:

    def __init__(self, weights, temperature) -> None:
        #weigts: ctc_seq, ctc_conv, distill
        self.CTC = nn.CTCLoss(blank=0, reduction='none')
        self.distll = SelfDistillLoss(temperature)
        self.weights = weights
    
    def __call__(self, conv_out, seq_out, length, target, target_length):
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
    
    # def _filter_nan(self, *losses):
    #     ret = []
    #     for loss in losses:
    #         if torch.all(torch.isinf(loss)).item():
    #             loss: torch.Tensor
    #             print('loss is inf')
    #             loss = torch.nan_to_num(loss, posinf=0.)
    #         ret.append(loss)
    #     return tuple(ret)

class SelfDistillLoss:
    def __init__(self, temperature) -> None:
        self.t = temperature
        
    def __call__(self, teacher, student):
        # seq: logits [t, n, c]
        T, N, C = teacher.shape
        assert (T, N, C) == student.shape
        teacher, student = teacher/self.t, student/self.t

        teacher = F.log_softmax(rearrange(teacher, 't n c -> (t n) c'), dim=-1)
        student = F.log_softmax(rearrange(student, 't n c -> (t n) c'), dim=-1)
        return F.kl_div(student, teacher.detach(), log_target=True, reduction='batchmean')

class HeatMapCTC(nn.Module):
    
    def __init__(self, color_range, weight, cfg_path, checkpoint, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        cfg = Config.fromfile(cfg_path)
        self.cfg = cfg
        self.color_range = color_range
        self.weight = weight

        self.register_buffer('std', torch.tensor(cfg.model.data_preprocessor.std))
        self.register_buffer('mean', torch.tensor(cfg.model.data_preprocessor.mean))
        self.vitpose = init_model(cfg, checkpoint)
        
        self.ctc_loss = nn.CTCLoss(reduction='none')
        self.l2_loss = nn.MSELoss(reduction='mean')
    
    def forward(self, backbone_out, input, input_length, target, target_length):
        #n 17 t h w
        loss = torch.tensor(0.)

        if self.weight[0] > 0.:
            heatmap = backbone_out.encoder_out.heatmap
            loss += self.l2_loss(heatmap, self._vitpose_predict(input)) * self.weight[0]
        if self.weight[1] > 0.:
            loss += self.ctc_loss(backbone_out.out, target, backbone_out.t_length.cpu().int(), target_length.cpu().int()).mean()* self.weights[1]

        return loss

    @torch.no_grad()
    def _vitpose_predict(self, x):
        #n c t h w
        _,_,T,_,_ = x.shpe
        x = rearrange(x, 'n c t h w -> (n t) c h w')
        x = self._data_preprocess(x)
        heatmap = self.vitpose(x, None)
        heatmap = rearrange(heatmap, '(n t) c h w -> n c t h w')
        return heatmap.detach()

    @staticmethod 
    def _data_preprocess(self, x):
        x = mapping_0_1(self.color_range, x)
        x = x * 255. #mapping to 0-255
        x = x.permute(0, 2, 3, 1)
        x = (x - self.mean) / self.std
        x = x.permute(0, 3, 1, 2)
        return x