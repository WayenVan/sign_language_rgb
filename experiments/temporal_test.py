import sys
sys.path.append('src')

from csi_sign_language.modules.tconv import TemporalConv
import torch

l = TemporalConv(128, 128)
n, t, c = 2, 79, 128
input = torch.ones(n, c, t)
lgt = torch.tensor([79, 30], dtype=torch.int32)
output = l(input, lgt)
print(output.size())
