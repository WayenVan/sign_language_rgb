import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import copy
from ...modules.bilstm import BiLSTMLayer
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

from csi_sign_language.utils.object import add_attributes


class BaseStream(nn.Module):

    def __init__(self, encoder, decoder, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.encoder = encoder
        self.rearrange = Rearrange('n c t -> t n c')
        self.decoder = decoder

    def forward(self, x, t_length):
        """

        :param x: [n, c, t, h, w]
        """
        encoder_out = self.encoder(x, t_length)

        x = self.rearrange(encoder_out['out'])
        t_length = encoder_out['t_length']
        decoder_out = self.decoder(x, t_length)

        return dict(
            out = decoder_out['out'],
            t_length = decoder_out['t_length'],
            encoder_out = encoder_out,
            decoder_out = decoder_out
        )
