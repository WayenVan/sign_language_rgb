import pdb
import copy
import torch
import collections
import torch.nn as nn
import torch.nn.functional as F
from .patch_merg import PatchMerge1D


class TemporalConv1D(nn.Module):
    def __init__(self, input_size, out_size, bottleneck_size, conv_type=2, pooling='max'):
        super(TemporalConv1D, self).__init__()
        self.input_size = input_size
        self.hidden_size = bottleneck_size
        self.out_size = out_size
        self.conv_type = conv_type

        self.kernel_size = self.conv_type

        modules = []
        for layer_idx, ks in enumerate(self.kernel_size):
            input_sz = self.input_size if layer_idx == 0 else self.hidden_size
            if ks[0] == 'P':
                assert len(modules) > 0, 'pooling layer should not be the first layer'
                assert len(modules) + 1 == len(self.kernel_size), 'pooling layer should not be the last layer'
                modules.append(self._build_pooling_layer(pooling, int(ks[1]), bottleneck_size))
            elif ks[0] == 'K':
                in_size = input_size if self._is_first_k(self.kernel_size, layer_idx) else bottleneck_size
                o_size = out_size if self._is_last_k(self.kernel_size, layer_idx) else bottleneck_size
                modules.append(
                    nn.Conv1d(in_size, o_size, kernel_size=int(ks[1]), stride=1, padding=int(ks[1])//2, bias=False)
                )
                modules.append(nn.BatchNorm1d(o_size))
                modules.append(nn.ReLU(inplace=True))
            else:
                raise NotImplementedError
        self.temporal_conv = nn.Sequential(*modules)
    
    @staticmethod
    def _build_pooling_layer(type, kernel, channels):
        if type == 'max':
            return nn.MaxPool1d(kernel, ceil_mode=False)
        elif type == 'mean':
            return nn.AvgPool1d(kernel, ceil_mode=False)
        elif type == 'patch':
            return PatchMerge1D(channels, channels, kernel)
        else:
            raise NotImplementedError

    @staticmethod
    def _is_first_k(kernels, idx):
        return all(kernels[i][0] != 'K' for i in range(idx))
    @staticmethod
    def _is_last_k(kernels, idx):
        return all(kernels[i][0] != 'K' for i in range(idx+1, len(kernels)))
    
    def get_kernel_size(self):
        return self.kernel_size

    def update_lgt(self, lgt):
        feat_len = lgt
        with torch.no_grad():
            for ks in self.kernel_size:
                if ks[0] == 'P':
                    feat_len = feat_len // int(ks[1])
                else:
                    feat_len = feat_len
        return feat_len

    def forward(self, frame_feat, lgt):
        """

        :param frame_feat: [n c t]
        :param lgt: [n]
        """
        visual_feat = self.temporal_conv(frame_feat)
        lgt = self.update_lgt(lgt)
        return visual_feat, lgt
