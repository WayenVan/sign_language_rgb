import sys
sys.path.append('src')

from csi_sign_language.modules.pooling import NormPoolTemp
import torch

p = NormPoolTemp(3, 1)
a = torch.ones(12, 2, 64, 24, 24)
b = p(a)
print(b)