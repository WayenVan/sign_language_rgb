import torch.nn as nn
import torch
import numpy
pool = nn.AvgPool1d(3, ceil_mode=True)

random = torch.rand((12, 128, 8))
print(pool(random).size())
print(torch.transpose(random, 0, -1).size())

def f(x: torch.Tensor):
    x = torch.ones((2,2))
    return x

f(random)
print(random.shape)