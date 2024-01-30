import torch.nn as nn
import torch
import numpy

for i in range(torch.cuda.device_count()):
    print(torch.cuda.get_device_properties(i).name)


pool = nn.AvgPool1d(3, ceil_mode=True)

random = torch.rand((12, 128, 8))
# random.cuda(0)
print(random.unsqueeze(0).shape)
