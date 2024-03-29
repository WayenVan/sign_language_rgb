import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
from einops import rearrange
from ..modules.resnet.resnet import *
from ..modules.tconv import *
from ..modules.bilstm import BiLSTMLayer
from torch.cuda.amp.autocast_mode import autocast
from .tadaconv.tadaconv import TAdaConv2d, RouteFuncMLP
from einops import rearrange

class TadaBlock(nn.Module):
    
    def __init__(self, in_channels, out_channels, ratio, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.ratio = ratio

        self.rf = RouteFuncMLP(in_channels, kernels=[3, 3], ratio=ratio)
        self.adaconv = TAdaConv2d(
            in_channels,
            out_channels,
            kernel_size=[1, 3, 3],
            stride=[1, 1, 1],
            padding=[0, 1, 1],
            bias=False
        )

        self.strided_avgpool = nn.AvgPool3d((3, 1, 1), stride=(1, 1, 1), padding=(1, 0, 0))
        self.bn = nn.BatchNorm3d(num_features=out_channels)
        self.bn_avg = nn.BatchNorm3d(num_features=out_channels)
        self.active = nn.GELU()
        
    
    def forward(self, x):
        N, C, T, H, W = x.size()
        x = self.adaconv(x, self.rf(x))
        x_branch1 = self.bn(x)
        
        x_branch2 = self.strided_avgpool(x)
        x_branch2 = self.bn_avg(x_branch2)
        return self.active(x_branch1 + x_branch2)
        
class ChannelSplitTada(nn.Module):
    
    def __init__(self, channels, split_ratio, tada_fr_ration, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.channels = channels
        self.split_ration = split_ratio
        self.tada_channels = channels//split_ratio
        self.tada = TadaBlock(self.tada_channels, self.tada_channels, tada_fr_ration)
    
    def forward(self, x):
        N, C, T, H, W = x.size()
        x_branch1 = x[:, :self.tada_channels]
        x_branch2 = x[:, self.tada_channels:]
        
        x_branch1 = self.tada(x_branch1)
        
        ret = torch.concat((x_branch1, x_branch2), dim=1)
        assert ret.size(1) == C
        
        return ret
    

class TadaResnet(nn.Module):
    
    def __init__(self,
                renset_type='resnet18',
                 *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.resnet: ResNet = resnet18(weights=ResNet18_Weights.DEFAULT)
        
        channels = [64, 128, 256, 512]
        self.tadablocks = nn.ModuleList()
        for i in range(4):
            self.tadablocks.append(
                ChannelSplitTada(channels[i], 2, tada_fr_ration=4)
            )
    
    def _forward_resnet_stem(self, x):
        #[b c h w]
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        return x
    
    def forward(self, x: torch.Tensor):
        T, N, C, H, W = x.size()
        x = rearrange(x, 't n c h w -> (t n) c h w')
        x = self._forward_resnet_stem(x)
        
        for i in range(4):
            x = getattr(self.resnet, f'layer{i+1}')(x)
            #[b c h w]
            x = rearrange(x, '(t n) c h w -> n c t h w', n=N)
            x = self.tadablocks[i](x)
            x = rearrange(x, 'n c t h w ->(t n) c h w')
        
        
        x = self.resnet.avgpool(x)
        x = torch.flatten(x, 1)
        
        x = rearrange(x, '(t n) c -> t n c', n=N)
        return x
        