import torch
import torch.nn as nn
import torch.nn.functional as F

from model.cdssm import CDSSM


class Siames(nn.Module):

    def __init__(self, out_size, **kwargs):
        super(Siames, self).__init__()
        self.out_size = out_size
        self.cdssm = CDSSM(out_size=out_size, **kwargs)

        self.combination_layer = nn.Linear(self.out_size[-1], 1)
        self.sm = nn.Sigmoid()

    def weight_init(self, init_foo):
        init_foo(self.combination_layer.weight)
        # nn.init.xavier_uniform_(self.combination_layer.bias)
        self.cdssm.weight_init(init_foo)

    def forward(self, batch_a, batch_b):
        vectors_a = self.cdssm.process_sentences(batch_a)
        vectors_b = self.cdssm.process_sentences(batch_b)

        out = self.sm(self.combination_layer(vectors_a * vectors_b)).squeeze()

        return out
