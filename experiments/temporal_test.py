import sys
sys.path.append('src')

from csi_sign_language.modules.tconv import TemporalConv, TemporalAveragePooling1D
import torch

l = TemporalConv(128, 128)
m = TemporalAveragePooling1D(l.get_kernel_size())
n, t, c = 2, 47, 128
input = torch.ones(n, c, t)
lgt = torch.tensor([79, 30], dtype=torch.int32)
output = l(input, lgt)
output2 = m(input)
print(output[0].size())
print(output2.size())
