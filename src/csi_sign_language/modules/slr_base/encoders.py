import torch
from typing import List
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import copy
from ...modules.bilstm import BiLSTMLayer
from ...modules.x3d import X3d
from ...modules.flownet2.models import FlowNet2SDConvDown, FlowNet2SD
from ...modules.tconv import TemporalConv1D
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from csi_sign_language.utils.object import add_attributes

class Conv_Pool_Proejction(nn.Module):

    def __init__(self, in_channels, out_channels, neck_channels, dropout=0.5, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        add_attributes(self, locals())
        self.drop = nn.Dropout(p=dropout, inplace=False)
        self.project1 = self.make_projection_layer(in_channels, neck_channels)
        self.project2 = self.make_projection_layer(neck_channels, neck_channels)
        self.linear = nn.Conv3d(neck_channels, out_channels, kernel_size=1, padding=0)
        self.spatial_pool = nn.AdaptiveAvgPool3d(output_size=(None, 1, 1))
        self.flatten = nn.Flatten(-3)

    @staticmethod
    def make_projection_layer(in_channels, out_channels):
        return nn.Sequential(
            nn.AvgPool3d((4, 3, 3), stride=(2, 1, 1), padding=1),
            nn.Conv3d(in_channels, out_channels,  kernel_size=1, stride=1),
            nn.BatchNorm3d(out_channels),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x, video_length):
        # n, c, t, h, w
        # n, l
        x = self.drop(x)
        x = self.project1(x)
        x = self.project2(x)
        x = self.spatial_pool(x)
        x = self.linear(x)
        x = self.flatten(x)
        
        video_length = video_length//2//2
        return x, video_length


class X3dEncoder(nn.Module):

    def __init__(self, out_channels, dropout, header_neck_channels=None, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        add_attributes(self, locals())
        if header_neck_channels is None:
            header_neck_channels = out_channels

        self.x3d = X3d()
        self.header = Conv_Pool_Proejction(self.x3d.x3d_out_channels, header_neck_channels, dropout)
    
    def forward(self, x, t_length):
        stem_out, stages_out = self.x3d(x)
        x = stages_out[-1]
        x, t_length = self.header(x, t_length)
        
        return dict(
            out=x,
            t_length=t_length,
            stem=stem_out,
            stages_out=stages_out
        )
        
class TemporalShift(nn.Module):
    
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    
    def forward(self, x, t_length):
        """

        :param x: n c t h w
        :param t_length: n
        :return: [s=2, n, c, t, h w], in dimension s: (x_t, x_t+1)
        """
        #n c t h w
        N = x.size(0)
        x_t0 = x
        #shift the time
        x_t1 = torch.cat([x[:, :, 1:], x[:, :, -1].unsqueeze(dim=2)], dim=2)

        #duplicate the last frame
        for n in range(N):
            length = int(t_length[n].item())
            x_t1[:, :, length-1] = x[:, :, length-1]
        #s=2, n, c, t, h, w
        return torch.stack([x_t0, x_t1], dim=0)
        

class FlowNetEncoder(nn.Module):
    
    def __init__(self, rgb_max, output_channels, flownet_checkpoint, is_freeze_flownet=False, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        add_attributes(self, locals())
        self.is_freeze_flownet=is_freeze_flownet
        #input size n c t h w
        self.shift = TemporalShift()
        self.flatten_tn = Rearrange('f n c t h w -> (n t) c f h w')
        self.flow_down_conv = FlowNet2SDConvDown(rgb_max)
        self.adap_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = Rearrange('n c h w -> n (c h w)')
        self.tconv = TemporalConv1D(self.flow_down_conv.output_channels, output_channels)
        
        checkpoint = torch.load(flownet_checkpoint)
        self.flow_down_conv.load_state_dict(checkpoint['state_dict'])
        
    def forward(self, x, t_length):
        #n c t h w
        N, C, T, H, W = x.shape
        x = self.shift(x, t_length)
        x = self.flatten_tn(x)
        flow_out = self.flow_down_conv(x)
        x = self.adap_pool(flow_out[-1])
        x = self.flatten(x)
        
        x = rearrange(x, '(n t) c -> n c t', n=N)
        x, t_length = self.tconv(x, t_length)
        return dict(
            out=x,
            t_length=t_length,
            flow=flow_out
        )
        
    def freeze_flownet(self):
        for parameter in self.flow_down_conv.parameters():
            parameter.requires_grad = False

    def train(self, mode=True):
        super().train(mode)
        if self.is_freeze_flownet:
            self.freeze_flownet()



class CnnFLowFusion(nn.Module):
    
    def __init__(self, x3d_channels, flow_channels, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.c_x3d = x3d_channels
        self.c_flow = flow_channels

        self.flow_conv = nn.Sequential(
            nn.Conv3d(2, flow_channels, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False),
            nn.BatchNorm3d(flow_channels),
            nn.LeakyReLU(inplace=True))
        self.joint_conv = nn.Sequential(
            nn.Conv3d(x3d_channels + flow_channels, x3d_channels, 1, 1),
            nn.BatchNorm3d(flow_channels),
            nn.LeakyReLU(inplace=True))
        
    def forward(self, features, flow, t_length):
        """

        :param features: [n c t h w]
        :param flow: [n d t h w]
        :param t_length: [n]
        """
        N = features.size(0)
        
        for idx, length in enumerate(t_length.cpu().tolist()):
            flow[idx][:, length:] = 0.
        
        flow = self.flow_conv(flow)
        cat_feature = torch.cat(features, flow, dim=1)
        fused_features = self.joint_conv(cat_feature)
        return fused_features + features
        

class X3DFlowDecoder(nn.Module):
    
    def __init__(self, out_channels, rgb_max, fusion_layers: List[List], flownet_checkpoint, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        
        if not (isinstance(fusion_layers, list) and isinstance(fusion_layers[0], (tuple, list))):
            raise Exception("fusion_layers format error")

        self.temporal_shift = TemporalShift()
        self.flatten_tn = Rearrange('f n c t h w -> (n t) c f h w')
        self.flow_net = FlowNet2SD(rgb_max)
        self.x3d = X3d()
        self.proejct = Conv_Pool_Proejction(self.x3d.x3d_out_channels, out_channels)

        self._create_fusion_layers(fusion_layers)
        self._load_flownet(flownet_checkpoint)
        
    def _load_flownet(self, checkpoint):
        checkpoint = torch.load(checkpoint)
        self.flow_down_conv.load_state_dict(checkpoint['state_dict'])
        
    
    def _create_fusion_layers(self, fusion_layers):
        self.fusion_layers = nn.ModuleList()
        for fusions in fusion_layers:
            self.fusion_layers.append(CnnFLowFusion(fusions[0], fusions[1]))
    
    def forward(self, x, t_length):
        #n c t h w
        N, C, T, H, W = x.shape
        flow_input = self.shift(x, t_length)
        flow_input = self.flatten_tn(flow_input)
        flow_out = self.flow_net(flow_input)
        flow_out = [rearrange(f, '(n t) d h w -> n d t h w') for f in flow_out]
        
        self._x3d_fusion(x, t_length)
        x, t_length = self.proejct(x, t_length)
        
        return dict(
            out=x,
            t_length=t_length,
            flow=flow_out
        )
        

    def _x3d_fusion(self, x, flows, t_length):
        """
        :param x: [n, c, t, h, w]
        """
        N, C, T, H, W = x.shape
        assert (H, W) == self.input_size_spatial, f"expect size {self.input_size_spatial}, got size ({H}, {W})"
        stages_out = []

        x = self.x3d.stem(x)
        x = self.fusion_layers[0](F.max_pool3d(x, (1, 2, 2)), flows[0], t_length)
        stem_out = x

        for idx, stage in enumerate(self.x3d.res_stages):
            x = stage(x)
            x = self.fusion_layers[idx+1](x, flows[idx+1], t_length)
            stages_out.append(x)
        
        return x
        