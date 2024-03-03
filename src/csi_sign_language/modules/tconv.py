import pdb
import copy
import torch
import collections
import torch.nn as nn
import torch.nn.functional as F


class TemporalConv1D(nn.Module):
    def __init__(self, input_size, hidden_size, conv_type=2):
        super(TemporalConv1D, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.conv_type = conv_type

        if self.conv_type == 0:
            self.kernel_size = ['K3']
        elif self.conv_type == 1:
            self.kernel_size = ['K5', "P2"]
        elif self.conv_type == 2:
            self.kernel_size = ['K5', "P2", 'K5', "P2"]
        elif self.conv_type == 3:
            self.kernel_size = ['K5', 'K5', 'P2']

        modules = []
        for layer_idx, ks in enumerate(self.kernel_size):
            input_sz = self.input_size if layer_idx == 0 else self.hidden_size
            if ks[0] == 'P':
                modules.append(nn.MaxPool1d(kernel_size=int(ks[1]), ceil_mode=False))
            elif ks[0] == 'K':
                modules.append(
                    nn.Conv1d(input_sz, self.hidden_size, kernel_size=int(ks[1]), stride=1, padding=0)
                )
                modules.append(nn.BatchNorm1d(self.hidden_size))
                modules.append(nn.ReLU(inplace=True))
        self.temporal_conv = nn.Sequential(*modules)
    
    def get_kernel_size(self):
        return self.kernel_size

    def update_lgt(self, lgt):
        feat_len = lgt
        with torch.no_grad():
            for ks in self.kernel_size:
                if ks[0] == 'P':
                    feat_len = feat_len // int(ks[1])
                else:
                    feat_len = feat_len - int(ks[1]) + 1
        return feat_len

    def forward(self, frame_feat, lgt):
        """

        :param frame_feat: [n c t]
        :param lgt: [n]
        """
        visual_feat = self.temporal_conv(frame_feat)
        lgt = self.update_lgt(lgt)
        return visual_feat, lgt
