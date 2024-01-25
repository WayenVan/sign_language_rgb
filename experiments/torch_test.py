import torch.nn as nn
import torch
import numpy

for i in range(torch.cuda.device_count()):
    print(torch.cuda.get_device_properties(i).name)


pool = nn.AvgPool1d(3, ceil_mode=True)

random = torch.rand((12, 128, 8))
random.cuda(0)
print(pool(random).size())
print(torch.transpose(random, 0, -1).size())

a = torch.tensor([0.1134, 0.0978, 0.0940])
print(a.sqrt())
print(a.device)
a = torch.tensor([0.], requires_grad=True).to(a.device)
a = torch.tensor
a.backward()
