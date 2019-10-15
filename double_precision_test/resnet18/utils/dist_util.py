import os
import socket
import torch as torch
import torch.distributed as dist
from torch.nn import Module
import os
import torch
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.utils import clip_grad

class DistModule(Module):
    def __init__(self, module):
        super(DistModule, self).__init__()
        self.module = module
        broadcast_params(self.module)

    def forward(self, *inputs, **kwargs):
        return self.module(*inputs, **kwargs)

    def train(self, mode=True):
        super(DistModule, self).train(mode)
        self.module.train(mode)

def sum_gradients(model):
    for param in model.parameters():
        if param.requires_grad:
            dist.all_reduce(param.grad.data)


def average_gradients(model):
    world_size = dist.get_world_size()
    for param in model.parameters():
        if param.requires_grad:
            dist.all_reduce(param.grad.data)
            param.grad.data /= world_size

def broadcast_params(model):
    for p in model.state_dict().values():
        dist.broadcast(p, 0)

def dist_init():
    raise NotImplementedError("Please use special code for your own distributed system")
    dist.init_process_group(backend='nccl')
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    return rank, world_size
