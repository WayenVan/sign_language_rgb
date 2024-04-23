import torch
from fast_ctc_decode import beam_search


posterios = torch.rand(10, 5)
poster = torch.nn.functional.softmax(posterios)

a = ['a', 'b', 'c', 'd', 'e']
alpha = [i for i in range(len(a))]

output = beam_search(poster.numpy(), beam_size=50, beam_cut_threshold=1e-5, alphabet=alpha)
print(output)
# help(beam_search)