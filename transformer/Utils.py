import torch
from torch.nn.parallel import DistributedDataParallel


class WrappedDataParallel(torch.nn.DataParallel):
    """
    Allow nn.DataParallel to call model's attributes.
    """
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)


class WrappedDistributedDataParallel(DistributedDataParallel):
    """
    Allow torch.distributed.DistributedDataParallel call model's attributes.
    """
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)
