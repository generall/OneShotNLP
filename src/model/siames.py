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

    def forward(self, batch_a, batch_b):
        vectors_a = self.cdssm.process_sentences(batch_a)
        vectors_b = self.cdssm.process_sentences(batch_b)

        #out = F.cosine_similarity(vectors_a, vectors_b)

        out = self.sm(self.combination_layer(vectors_a * vectors_b))

        if self.debug:
            self.params = {
                'max_value': torch.max(vectors_a).item(),
                'avg_value': torch.mean(vectors_a[vectors_a > 0.001]).item(),
            }

        return out
