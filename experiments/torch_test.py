import torch.nn as nn
import torch
import numpy
pool = nn.AvgPool1d(3, ceil_mode=True)

random = torch.rand((12, 128, 8))
print(pool(random).size())

a = [1, 2, 3, 4, 5]
print(a[::2])


a = numpy.random.rand(3, 3)
a[0] = [1., 1., 1]
print(a)
