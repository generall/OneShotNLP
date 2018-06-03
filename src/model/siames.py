import torch
import torch.nn as nn
import torch.nn.functional as F

from model.cdssm import CDSSM


class Siames(nn.Module):

    def __init__(self, debug=False, **kwargs):
        super(Siames, self).__init__()
        self.cdssm = CDSSM(**kwargs)
        self.debug = debug  # If true - perform saving statistics
        self.params = {}

    def forward(self, batch_a, batch_b):
        vectors_a = self.cdssm.process_sentences(batch_a)
        vectors_b = self.cdssm.process_sentences(batch_b)

        distances = F.cosine_similarity(vectors_a, vectors_b)

        if self.debug:
            self.params = {
                'max_value': torch.max(vectors_a).item(),
                'avg_value': torch.mean(vectors_a[vectors_a > 0.001]).item(),
            }

        return distances
