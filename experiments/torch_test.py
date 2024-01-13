import torch.nn as nn
import torch
import numpy
pool = nn.AvgPool1d(3, ceil_mode=True)

random = torch.rand((12, 128, 8))
print(pool(random).size())
print(torch.transpose(random, 0, -1).size())

import sys
sys.path.append('src')
from csi_sign_language.utils.lr_scheduler import WarmUpLr

wlr = WarmUpLr(1e-9, 1, 2, min_lr=1e-6, decay_factor=0.95)


for i in range(100):
    print(wlr(i))