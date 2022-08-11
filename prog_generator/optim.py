"""
author: Adnen Abdessaied
maintainer: "Adnen Abdessaied"
website: adnenabdessaied.de
version: 1.0.1
"""

# --------------------------------------------------------
# adapted from https://github.com/MILVLG/mcan-vqa/blob/master/core/model/optim.py
# --------------------------------------------------------

import torch
import torch.optim as Optim


class WarmupOptimizer(object):
    def __init__(self, lr_base, optimizer, data_size, batch_size):
        self.optimizer = optimizer
        self._step = 0
        self.lr_base = lr_base
        self._rate = 0
        self.data_size = data_size
        self.batch_size = batch_size

    def step(self):
        self._step += 1

        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate

        self.optimizer.step()

    def zero_grad(self):
        self.optimizer.zero_grad()

    def rate(self, step=None):
        if step is None:
            step = self._step

        if step <= int(self.data_size / self.batch_size * 1):
            r = self.lr_base * 1/2.
        else:
            r = self.lr_base

        return r


def get_optim(opts, model, data_size, lr_base=None):
    if lr_base is None:
        lr_base = opts.lr

    if opts.optim == 'adam':
        optim = Optim.Adam(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=0,
                betas=opts.betas,
                eps=opts.eps,

            )
    elif opts.optim == 'rmsprop':
        optim = Optim.RMSprop(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=0,
                eps=opts.eps,
                weight_decay=opts.weight_decay
            )
    else:
        raise ValueError('{} optimizer is not supported'.fromat(opts.optim))
    return WarmupOptimizer(
        lr_base,
        optim,
        data_size,
        opts.batch_size
    )

def adjust_lr(optim, decay_r):
    optim.lr_base *= decay_r
