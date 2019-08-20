import torch

from utils.pad import push_zeros_right, calc_pos

# a = torch.tensor([[0, 2, 3, 4],[0, 0, 1, 5],[2, 3, 0, 1]])
#
# print(a)
# print(push_zeros_right(a))
#
#
# x = (torch.rand(10, 20) * 5).round()
#
# print(x)
# print(calc_pos(x))
#
# print(x.shape)
# print(calc_pos(x).shape)



# t = torch.randn(3, 4, 5)
# i = (torch.rand(3) * 4).long()
#
# print(t[0])
# print(t.shape)
# print(i[0])
# print(i.shape)
#
# result = batched_index_select(t, 1, i)
# print(result[0])
# print(result.shape)