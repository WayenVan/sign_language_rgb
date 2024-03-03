import torch
from torch import nn
from ..trainsformer import TransformerEncoder
from ..tconv import TemporalConv1D
from ...utils.object import add_attributes
from einops import rearrange, repeat

class SpatialAuxHeader(nn.Module):
    
    def __init__(self, input_size, hidden_size, vocab_size, d_feedforward, n_head, n_layers, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        add_attributes(self, locals())
        self.adaptive_pool = nn.AdaptiveAvgPool3d((-1, 5, 5))
        self.tconv = TemporalConv1D(input_size, hidden_size)
        self.sequence_model = TransformerEncoder(hidden_size, d_feedforward, n_head, n_layers)
        self.fc = nn.Linear(hidden_size, vocab_size)
    
    
    def forward(self, video_features, video_length):
        """
        :param video_features: [n c t h w]
        :param video_length: [n]
        """
        x = self.adaptive_pool(video_features)
        #n, c, t, 5, 5
        video_length = repeat(video_length,'n -> n h w', h=5, w=5)
        x = rearrange(x, 'n c t h w -> (n h w) c t')
        video_length = rearrange(video_length, 'n h w -> (n h w)')
        
        x, video_length= self.tconv(x, video_length)
        #[(n h w) c t]
        
        x = rearrange(x, 'a c t -> t a c')
        x = self.sequence_model(x, video_length)
        x = self.fc(x)
        #[t (n h w) vocab]
        x = rearrange(x, 't (n h w) d -> t h w n d')
        video_length = repeat(video_length, 'n -> n h w', h=5, w=5)

        #t h w n d, n h w
        return x, video_length

        
        
class TemporalAuxHeader(nn.Module):
    
    def __init__(self, input_size, hidden_size, vocab_size, d_feedforward, n_head, n_layers, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        add_attributes(self, locals())
        self.adaptive_pool = nn.AdaptiveAvgPool3d((-1, 5, 5))
        self.spatial_conv = nn.Conv3d(input_size, hidden_size, kernel_size=(1, 5, 5), stride=(1, 1, 1), padding=(0, 1, 1))
        self.flatten = nn.Flatten(-2)
        self.t_pool = nn.AvgPool1d((4))
        self.sequence_model = TransformerEncoder(hidden_size, d_feedforward, n_head, n_layers)
        self.fc = nn.Linear(hidden_size, vocab_size)
    
    
    def forward(self, video_features, video_length):
        """
        :param video_features: [n c t h w], []
        :param video_length: [n]
        """

        x = self.adaptive_pool(video_features)
        #n, c, t, 5, 5
        x = self.spatial_conv(x)
        x = self.flatten(x)
        x = self.t_pool(x)
        with torch.no_grad():
            video_length = video_length//4
        
        x = rearrange(x, 'n c t -> n t c')
        x = self.sequence_model(x)
        x = self.fc(x)
        return x, video_length