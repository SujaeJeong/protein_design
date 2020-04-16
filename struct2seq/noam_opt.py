import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader

class NoamOpt:
    "Optim wrapper that implements rate."
    def __init__(self, args, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = args.warmup
        self.factor = 2
        self.model_size = args.hidden
        self._rate = 0
        
    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()
        
    def rate(self, step = None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * \
            (self.model_size ** (-0.5) *
            min(step ** (-0.5), step * self.warmup ** (-1.5)))

    def zero_grad(self):
        self.optimizer.zero_grad()
        
def get_std_opt(parameters, args):
    return NoamOpt(
        args, torch.optim.Adam(parameters, lr=0, betas=(0.9, 0.98), eps=1e-9)
    )