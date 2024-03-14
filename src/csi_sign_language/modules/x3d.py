import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import copy
from einops import rearrange, repeat

from csi_sign_language.utils.object import add_attributes

class X3d(nn.Module):

    def __init__(self, x3d_type='x3d_s', *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        add_attributes(self, locals())

        self.input_size_spatial = self.x3d_spec[x3d_type]['input_shape']
        self.x3d_out_channels, self.conv_neck_channels = self.x3d_spec[x3d_type]['channels']

        x3d = torch.hub.load('facebookresearch/pytorchvideo', x3d_type, pretrained=True)
        self.move_x3d_layers(x3d)
        del x3d
        self.spec = self.x3d_spec[x3d_type]

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
        self.res_stages = nn.ModuleList(
            [copy.deepcopy(block) for block in blocks[1:-1]]
            )

    def forward(self, x):
        """
        :param x: [n, c, t, h, w]
        """
        N, C, T, H, W = x.shape
        # assert (H, W) == self.input_size_spatial, f"expect size {self.input_size_spatial}, got size ({H}, {W})"
        stages_out = []

        x = self.stem(x)
        stem_out = x

        for stage in self.res_stages:
            x = stage(x)
            stages_out.append(x)

        return stem_out, stages_out