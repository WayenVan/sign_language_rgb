import torch

loss = torch.nn.CTCLoss()
input = torch.rand([14, 2, 100]).log_softmax(-1)
target = torch.zeros([2, 7])
input_l = torch.tensor([14, 14])
target_l = torch.tensor([7, 7])
a = loss(input, target, input_l, target_l)
print(a)