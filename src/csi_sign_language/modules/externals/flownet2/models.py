import torch
from types import SimpleNamespace
import torch.nn as nn
from torch.nn import init

import math
import numpy as np

from .networks.resample2d_package.resample2d import Resample2d
from .networks.channelnorm_package.channelnorm import ChannelNorm

from .networks import FlowNetC
from .networks import FlowNetS
from .networks import FlowNetSD
from .networks import FlowNetFusion

from .networks.submodules import *

__all__ = ['FlowNet2SDConvDown', 'FlowNet2SD']

class FlowNet2SDConvDown(FlowNetSD.FlowNetSD):
    def __init__(self, color_range, batchNorm=False, div_flow=20):
        super(FlowNet2SDConvDown,self).__init__(None, batchNorm=batchNorm)
        self.color_range = color_range
        self.div_flow = div_flow
        
    def rescale(self, x):
        return self.color_range[0] + (x - self.color_range[0]) / (self.color_range[1] - self.color_range[0])

    def forward(self, inputs):

        inputs = self.rescale(inputs)
        rgb_mean = inputs.contiguous().view(inputs.size()[:2]+(-1,)).mean(dim=-1).view(inputs.size()[:2] + (1,1,1,))
        x = (inputs - rgb_mean) / 1.0 

        x = torch.cat( (x[:,:,0,:,:], x[:,:,1,:,:]), dim = 1)

        out_conv0 = self.conv0(x)
        out_conv1 = self.conv1_1(self.conv1(out_conv0))
        out_conv2 = self.conv2_1(self.conv2(out_conv1))

        out_conv3 = self.conv3_1(self.conv3(out_conv2))
        out_conv4 = self.conv4_1(self.conv4(out_conv3))
        out_conv5 = self.conv5_1(self.conv5(out_conv4))
        out_conv6 = self.conv6_1(self.conv6(out_conv5))

        return (
            out_conv0,
            out_conv1,
            out_conv2,
            out_conv3,
            out_conv4,
            out_conv5,
            out_conv6,
        )

        # flow6       = self.predict_flow6(out_conv6)
        # flow6_up    = self.upsampled_flow6_to_5(flow6)
        # out_deconv5 = self.deconv5(out_conv6)
        
        # concat5 = torch.cat((out_conv5,out_deconv5,flow6_up),1)
        # out_interconv5 = self.inter_conv5(concat5)
        # flow5       = self.predict_flow5(out_interconv5)

        # flow5_up    = self.upsampled_flow5_to_4(flow5)
        # out_deconv4 = self.deconv4(concat5)
        
        # concat4 = torch.cat((out_conv4,out_deconv4,flow5_up),1)
        # out_interconv4 = self.inter_conv4(concat4)
        # flow4       = self.predict_flow4(out_interconv4)
        # flow4_up    = self.upsampled_flow4_to_3(flow4)
        # out_deconv3 = self.deconv3(concat4)
        
        # concat3 = torch.cat((out_conv3,out_deconv3,flow4_up),1)
        # out_interconv3 = self.inter_conv3(concat3)
        # flow3       = self.predict_flow3(out_interconv3)
        # flow3_up    = self.upsampled_flow3_to_2(flow3)
        # out_deconv2 = self.deconv2(concat3)

        # concat2 = torch.cat((out_conv2,out_deconv2,flow3_up),1)
        # out_interconv2 = self.inter_conv2(concat2)
        # flow2 = self.predict_flow2(out_interconv2)

        # if self.training:
        #     return flow2,flow3,flow4,flow5,flow6
        # else:
        #     return self.upsample1(flow2*self.div_flow)

class FlowNet2SD(FlowNetSD.FlowNetSD):
    def __init__(self, color_range, batchNorm=False, div_flow=20):
        super(FlowNet2SD,self).__init__(None, batchNorm=batchNorm)
        self.color_range = color_range
        self.div_flow = div_flow
        
    def rescale(self, x):
        return self.color_range[0] + (x - self.color_range[0]) / (self.color_range[1] - self.color_range[0])

    def forward(self, inputs):
        inputs = self.rescale(inputs)
        rgb_mean = inputs.contiguous().view(inputs.size()[:2]+(-1,)).mean(dim=-1).view(inputs.size()[:2] + (1,1,1,))
        x = (inputs - rgb_mean) / 1.0 

        x = torch.cat( (x[:,:,0,:,:], x[:,:,1,:,:]), dim = 1)

        out_conv0 = self.conv0(x)
        out_conv1 = self.conv1_1(self.conv1(out_conv0))
        out_conv2 = self.conv2_1(self.conv2(out_conv1))

        out_conv3 = self.conv3_1(self.conv3(out_conv2))
        out_conv4 = self.conv4_1(self.conv4(out_conv3))
        out_conv5 = self.conv5_1(self.conv5(out_conv4))
        out_conv6 = self.conv6_1(self.conv6(out_conv5))

        flow6       = self.predict_flow6(out_conv6)
        flow6_up    = self.upsampled_flow6_to_5(flow6)
        out_deconv5 = self.deconv5(out_conv6)
        
        concat5 = torch.cat((out_conv5,out_deconv5,flow6_up),1)
        out_interconv5 = self.inter_conv5(concat5)
        flow5       = self.predict_flow5(out_interconv5)

        flow5_up    = self.upsampled_flow5_to_4(flow5)
        out_deconv4 = self.deconv4(concat5)
        
        concat4 = torch.cat((out_conv4,out_deconv4,flow5_up),1)
        out_interconv4 = self.inter_conv4(concat4)
        flow4       = self.predict_flow4(out_interconv4)
        flow4_up    = self.upsampled_flow4_to_3(flow4)
        out_deconv3 = self.deconv3(concat4)
        
        concat3 = torch.cat((out_conv3,out_deconv3,flow4_up),1)
        out_interconv3 = self.inter_conv3(concat3)
        flow3       = self.predict_flow3(out_interconv3)
        flow3_up    = self.upsampled_flow3_to_2(flow3)
        out_deconv2 = self.deconv2(concat3)

        concat2 = torch.cat((out_conv2,out_deconv2,flow3_up),1)
        out_interconv2 = self.inter_conv2(concat2)
        flow2 = self.predict_flow2(out_interconv2)

        return self.upsample1(flow2*self.div_flow), flow2,flow3,flow4,flow5,flow6
        #     return self.upsample1(flow2*self.div_flow)
        
