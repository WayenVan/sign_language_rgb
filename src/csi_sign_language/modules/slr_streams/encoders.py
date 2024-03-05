import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import copy
from ...modules.bilstm import BiLSTMLayer
from ...modules.x3d import X3d
from einops import rearrange, repeat

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
            nn.BatchNorm3d(out_channels)
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