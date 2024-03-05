
import torch.nn as nn
from ...modules.resnet.resnet import *
from ...modules.tconv import *
from ...modules.transformer import TransformerEncoder

class TransformerDecoder(nn.Module):
    
    def __init__(self, n_class, d_model, n_heads, n_layers, d_feedforward, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.tf = TransformerEncoder(d_model, d_feedforward, n_heads, n_layers)
        self.header = nn.Linear(d_model, n_class)
        
    def forward(self, x, seq_length):
        x = self.tf(x, seq_length)
        seq_out = x
        x = self.header(x)
        return dict(
            out = x,
            seq_out = seq_out
        )