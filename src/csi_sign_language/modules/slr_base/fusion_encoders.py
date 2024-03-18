import torch
from torch import nn
from torch.nn import functional as F
from einops import einsum

from einops.layers.torch import Rearrange
from .encoders import TemporalShift

class Fusion(nn.Module):

    def __init__(self, x3d_channels, flow_propotion, corr_channels_in, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.c_x3d = x3d_channels
        self.c_flow = x3d_channels * flow_propotion 
        self.c_corr = x3d_channels - self.c_flow
        
        self.c_corr_in =  corr_channels_in
        
        self.bn_x3d = nn.BatchNorm3d(self.c_x3d)

        self.flow_conv = nn.Sequential(
            nn.Conv3d(2, self.c_flow, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False),
            nn.BatchNorm3d(self.c_flow),
            nn.LeakyReLU(inplace=True))

        self.corr_conv = nn.Sequential(
            nn.Conv3d(self.c_corr_in, self.c_corr, 1, 1),
            nn.BatchNorm3d(x3d_channels),
            nn.LeakyReLU(inplace=True))
        
        self.weights = nn.Parameter(torch.random(x3d_channels), requires_grad=True)
        
    def forward(self, features, flow, corr, t_length):
        """

        :param features: [n c t h w]
        :param flow: [n d t h w]
        :param t_length: [n]
        """
        N = features.size(0)
        
        mask = torch.ones_like(flow)
        for idx, length in enumerate(t_length.cpu().tolist()):
            mask[idx][:, length:] = 0.
        flow = flow * mask
        
        flow = self.flow_conv(flow)
        corr = self.corr_conv(corr)
        cat_feature = torch.cat([corr, flow], dim=1)
        weights = F.sigmoid(self.weights)

        return einsum(features, weights, 'n c t h w, c -> n c t h w') + einsum(cat_feature, 1. - weights, 'n c t h w, c -> n c t h w')
    
    
class X3DFlowEncoder(nn.Module):
    
    def __init__(self, out_channels, color_range, fusion_layers: List[List], flownet_checkpoint, freeze_flownet=True, freeze_x3d=False, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        add_attributes(self, locals())

        self.temporal_shift = TemporalShift()
        self.flatten_tn = Rearrange('f n c t h w -> (n t) c f h w')
        self.flownet = FlowNet2SD(color_range)
        self.x3d = X3d()
        self.proejct = Conv_Pool_Proejction(self.x3d.x3d_out_channels, out_channels, neck_channels=out_channels*2)

        self._create_fusion_layers(fusion_layers)
        self._load_flownet(flownet_checkpoint)
        
    def _load_flownet(self, checkpoint):
        checkpoint = torch.load(checkpoint)
        self.flownet.load_state_dict(checkpoint['state_dict'])
        
    
    def _create_fusion_layers(self, fusion_layers):
        self.fusion_layers = nn.ModuleList()
        for fusions in fusion_layers:
            self.fusion_layers.append(Fusion(*fusions))
    
    def forward(self, x, t_length):
        #n c t h w
        N, C, T, H, W = x.shape
        flow_input = self.temporal_shift(x, t_length)
        flow_input = self.flatten_tn(flow_input)
        flow_out = self.flownet(flow_input)
        flow_out = [rearrange(f, '(n t) d h w -> n d t h w', n=N) for f in flow_out]
        
        x = self._x3d_fusion(x, flow_out, t_length)
        x, t_length = self.proejct(x, t_length)
        
        return dict(
            out=x,
            t_length=t_length,
            flow=flow_out
        )

    def train(self, mode=True):
        super().train(mode)

        for p in self.flownet.parameters():
                p.requires_grad = not self.freeze_flownet

        for p in self.x3d.parameters():
                p.requires_grad = not self.freeze_x3d
    

    def _x3d_fusion(self, x, flows, corrs, t_length):
        """
        :param x: [n, c, t, h, w]
        """
        N, C, T, H, W = x.shape
        stages_out = []

        x = self.x3d.stem(x)

        for idx, stage in enumerate(self.x3d.res_stages):
            x = stage(x)
            x = self.fusion_layers[idx](x, flows[idx], corrs[idx], t_length)
            stages_out.append(x)
        
        return x
        

