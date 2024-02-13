import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import copy
from ...modules.bilstm import BiLSTMLayer
from einops import rearrange

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


class X3dLSTM(nn.Module):

    def __init__(self, d_model, n_class, n_layers, x3d_type='x3d_s', dropout=0.5, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        add_attributes(self, locals())
        self.input_size_spatial = self.x3d_spec[x3d_type]['input_shape']
        self.x3d_out_channels, self.conv_neck_channels = self.x3d_spec[x3d_type]['channels']

        x3d = torch.hub.load('facebookresearch/pytorchvideo', x3d_type, pretrained=True)
        self.move_x3d_layers(x3d)
        self.projection = Conv_Pool_Proejction(self.x3d_out_channels, d_model, self.conv_neck_channels, dropout=dropout)

        self.rnn = BiLSTMLayer(d_model, hidden_size=d_model, num_layers=n_layers, bidirectional=True)
        self.fc_conv = nn.Linear(d_model, n_class)
        self.fc = nn.Linear(d_model, n_class)
    
    @property
    def x3d_spec(self):
        return dict(
            x3d_m=dict(
                channels=(192, 432),
                input_shape=(224, 224)
                ),
            x3d_s=dict(
                channels=(192, 432),
                input_shape=(160, 160)
                ),
        )

    def move_x3d_layers(self, x3d: nn.Module):
        blocks = x3d.blocks
        self.stem = copy.deepcopy(blocks[0])
        self.res_stages = nn.ModuleList(blocks[1:-1])

    def forward(self, x, video_length):
        """

        :param x: [t, n, c, h, w]
        """
        T, N, C, H, W = x.shape
        assert (H, W) == self.input_size_spatial, f"expect size {self.input_size_spatial}, got size ({H}, {W})"

        x = rearrange(x, 't n c h w -> n c t h w')
        x = self.stem(x)
        for stage in self.res_stages:
            x = stage(x)
        x, video_length= self.projection(x, video_length)
        x = rearrange(x, 'n c t -> t n c')
        conv_out = self.fc_conv(x)

        x = self.rnn(x, video_length)['predictions']
        seq_out = self.fc(x)

        return dict(
            seq_out = seq_out,
            conv_out = conv_out,
            video_length = video_length
        )
