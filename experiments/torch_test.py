import torch.nn as nn
import torch
import numpy
pool = nn.AvgPool1d(3, ceil_mode=True)

random = torch.rand((12, 128, 8))
print(pool(random).size())
print(torch.transpose(random, 0, -1).size())

print(random[::3].size())