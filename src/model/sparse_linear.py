import itertools

import torch
from torch import nn as nn


def pairwise(iterable):
    """
    s -> (s0,s1), (s1,s2), (s2, s3), ...

    :param iterable:
    :return:
    """
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)

class SparseLinear(nn.Linear):
    def __init__(self, dict_size, out_features, bias=True):
        super(SparseLinear, self).__init__(in_features=dict_size, out_features=out_features, bias=bias)

    def forward(self, inpt):
        res = torch.index_select(self.weight.t(), 0, inpt.view(-1))
        res = res.view(-1, inpt.shape[-1], self.out_features)
        res = res.sum(dim=1)
        res = res.view(*inpt.shape[:-1], self.out_features)

        if self.bias is not None:
            res = res + self.bias
        return res