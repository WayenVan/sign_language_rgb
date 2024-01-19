import torch.nn as nn
import torch
import numpy
pool = nn.AvgPool1d(3, ceil_mode=True)

random = torch.rand((12, 128, 8))
print(pool(random).size())
print(torch.transpose(random, 0, -1).size())

a = torch.tensor([0.1134, 0.0978, 0.0940])
print(a.sqrt())

def f():
    b = 0
    return (b+1+1)

a = f()
print(a[0])