import torch
from torch import nn
from mmengine.config import Config
from mmpose.apis import init_model
from csi_sign_language.utils.data import mapping_0_1
from einops import rearrange


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