import torch.nn as nn
import torch
import numpy
pool = nn.AvgPool1d(3, ceil_mode=True)

random = torch.rand((12, 128, 8))
print(pool(random).size())
print(torch.transpose(random, 0, -1).size())

a = [1, 2, 3, 4, 5]
print(a[::2])
a = torch.tensor([1, 2, 3, 0 ,0], dtype=torch.bool)
print(~a)
