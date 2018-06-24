import torch
import torch.nn as nn
import torch.nn.functional as F

from model.cdssm import CDSSM


class Siames(nn.Module):

    def __init__(self, out_size, debug=False, **kwargs):
        super(Siames, self).__init__()
        self.out_size = out_size
        self.cdssm = CDSSM(out_size=out_size, **kwargs)
        self.debug = debug  # If true - perform saving statistics
        self.params = {}
        self.combination_layer = nn.Linear(self.out_size[-1], 2)
        self.sm = nn.Softmax(dim=1)

    def weight_init(self, init_foo):
        init_foo(self.combination_layer.weight)
        # nn.init.xavier_uniform_(self.combination_layer.bias)
        self.cdssm.weight_init(init_foo)

    def forward(self, batch_a, batch_b):
        vectors_a = self.cdssm.process_sentences(batch_a)
        vectors_b = self.cdssm.process_sentences(batch_b)

        out = self.sm(self.combination_layer(vectors_a * vectors_b))

        if self.debug:
            self.params = {
                'max_value': torch.max(vectors_a).item(),
                'avg_value': torch.mean(vectors_a).item(),
            }

        return out
