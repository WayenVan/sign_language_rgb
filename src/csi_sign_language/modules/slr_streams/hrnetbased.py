import torch.nn as nn
from einops import rearrange
from ...modules.resnet.resnet import *
from ...modules.tconv import *
from ...modules.bilstm import BiLSTMLayer
from ...modules.sthrnet import STHrnet

class HrnetLSTM(nn.Module):
    
    def __init__(
        self,
        n_class,
        n_layers,
        hr_checkpoint,
        d_model = 512,
        if_freeze = True,
        norm_eval = False
        ) -> None:
        super().__init__()
        
        self.norm_eval = norm_eval
        self.if_freeze = if_freeze

        self.sthrnet = STHrnet(hr_checkpoint, if_freeze)
        d_model = self.sthrnet.outchannel
        
        self.tconv = TemporalConv(d_model, 2*d_model)
        self.rnn = BiLSTMLayer(2*d_model, hidden_size=2*d_model, num_layers=n_layers, bidirectional=True)
        self.fc_conv = nn.Linear(2*d_model, n_class)
        self.fc = nn.Linear(2*d_model, n_class)
    
    def train(self, mode: bool = True):
        super().train(mode)
        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                    m.eval()
    
    
    def forward(self, x, video_length):
        """
        :param x: [t, n, c, h, w]
        :param video_length: [n]
        """
        batch_size = x.size(dim=1)
        x, video_length = self.sthrnet(x, video_length)
        x = rearrange(x, 't n c -> n c t')
        x, video_length = self.tconv(x, video_length)
        x = rearrange(x, 'n c t -> t n c')
        conv_out = self.fc_conv(x)
        
        x = self.rnn(x, video_length)['predictions']
        x = self.fc(x)

        return dict(
            seq_out=x,
            conv_out=conv_out,
            video_length=video_length 
        )