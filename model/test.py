import torch

from utils.pad import push_zeros_right

a = torch.tensor([[0, 2, 3, 4],[0, 0, 1, 5],[2, 3, 0, 1]])

print(a)
print(push_zeros_right(a))