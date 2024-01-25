import pdb
import copy
from . import utils
import torch
import types
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from .modules.criterions import SeqKD
from .modules import BiLSTMLayer, TemporalConv


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class NormLinear(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(NormLinear, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(in_dim, out_dim))
        nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain('relu'))

    def forward(self, x):
        outputs = torch.matmul(x, F.normalize(self.weight, dim=0))
        return outputs


class SLRModel(nn.Module):
    def __init__(
            self, num_classes, c2d_type, conv_type, vocab, use_bn=False,
            hidden_size=1024, loss_weights=None,
            weight_norm=True, share_classifier=True
    ):
        gloss_dict = dict((key, [idx, 1]) for idx, key in enumerate(vocab.get_itos()))
        
        super(SLRModel, self).__init__()
        self.decoder = None
        self.loss = dict()
        self.criterion_init()
        self.num_classes = num_classes
        self.loss_weights = loss_weights
        self.conv2d = getattr(models, c2d_type)(pretrained=True)
        self.conv2d.fc = Identity()
        self.conv1d = TemporalConv(input_size=512,
                                   hidden_size=hidden_size,
                                   conv_type=conv_type,
                                   use_bn=use_bn,
                                   num_classes=num_classes)
        self.decoder = utils.Decode(gloss_dict, num_classes, 'beam')
        self.temporal_model = BiLSTMLayer(rnn_type='LSTM', input_size=hidden_size, hidden_size=hidden_size,
                                          num_layers=2, bidirectional=True)
        if weight_norm:
            self.classifier = NormLinear(hidden_size, self.num_classes)
            self.conv1d.fc = NormLinear(hidden_size, self.num_classes)
        else:
            self.classifier = nn.Linear(hidden_size, self.num_classes)
            self.conv1d.fc = nn.Linear(hidden_size, self.num_classes)
        if share_classifier:
            self.conv1d.fc = self.classifier
        # self.register_backward_hook(self.backward_hook)

    # def backward_hook(self, module, grad_input, grad_output):
    #     for g in grad_input:
    #         g[g != g] = 0

    def masked_bn(self, inputs, len_x):
        def pad(tensor, length):
            return torch.cat([tensor, tensor.new(length - tensor.size(0), *tensor.size()[1:]).zero_()])

        x = torch.cat([inputs[len_x[0] * idx:len_x[0] * idx + lgt] for idx, lgt in enumerate(len_x)])
        x = self.conv2d(x)
        x = torch.cat([pad(x[sum(len_x[:idx]):sum(len_x[:idx + 1])], len_x[0])
                       for idx, lgt in enumerate(len_x)])
        return x

    def forward(self, x: torch.Tensor, len_x):
        # t, b, c, h, w
        assert len(x.shape) == 5
        x = x.permute(1, 0, 2, 3, 4)

        # videos
        batch, temp, channel, height, width = x.shape
        inputs = x.reshape(batch * temp, channel, height, width)
        # framewise = self.masked_bn(inputs, len_x)
        framewise = self.conv2d(inputs)
        framewise = framewise.reshape(batch, temp, -1).transpose(1, 2)

        conv1d_outputs = self.conv1d(framewise, len_x)
        # x: T, B, C
        x = conv1d_outputs['visual_feat']
        lgt = conv1d_outputs['feat_len']
        tm_outputs = self.temporal_model(x, lgt)
        outputs = self.classifier(tm_outputs['predictions'])
        pred = self.decoder.decode(outputs, lgt, batch_first=False, probs=False)
        conv_pred = self.decoder.decode(conv1d_outputs['conv_logits'], lgt, batch_first=False, probs=False)

        return {
            "framewise_features": framewise,
            "visual_features": x,
            "video_length": lgt,
            "conv_out": conv1d_outputs['conv_logits'],
            "seq_out": outputs,
            "conv_out_label": [[v[0] for v in item] for item in conv_pred],
            "seq_out_label": [[v[0] for v in item] for item in pred],
        }

    def inference(self, *args, **kwargs):
        with torch.no_grad():
            ret = self.forward(*args, **kwargs)
        return ret['seq_out_label']

    def criterion(self, ret_dict, label, label_lgt):
        loss = 0
        for k, weight in self.loss_weights.items():
            if k == 'ConvCTC':
                loss += weight * \
                    self.loss['CTCLoss'](ret_dict["conv_out"].log_softmax(-1),
                                                      label.cpu().int(), ret_dict["video-length"].cpu().int(),
                                                      label_lgt.cpu().int()).mean()
            elif k == 'SeqCTC':
                loss += weight * \
                    self.loss['CTCLoss'](ret_dict["seq_out"].log_softmax(-1),
                                                      label.cpu().int(), ret_dict["video_length"].cpu().int(),
                                                      label_lgt.cpu().int()).mean()
            elif k == 'Dist':
                loss += weight * self.loss['distillation'](ret_dict["conv_out"],
                                                           ret_dict["seq_out"].detach(),
                                                           use_blank=False)
        if np.isinf(loss.item()) or np.isnan(loss.item()):
            loss = torch.nan_to_num(loss, nan=0., posinf=0., neginf=0.)

        return loss

    def criterion_init(self):
        self.loss['CTCLoss'] = torch.nn.CTCLoss(reduction='none', zero_infinity=False)
        self.loss['distillation'] = SeqKD(T=8)
        return self.loss
