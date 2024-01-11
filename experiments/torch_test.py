import torch.nn as nn
import torch

pool = nn.AvgPool1d(3, ceil_mode=True)

random = torch.rand((12, 128, 8))
print(pool(random).size())

a = [1, 2, 3, 4, 5]
print(a[::2])

class A:
    lr: int
    sdfs: str
    
a = A()
print(a.lr)