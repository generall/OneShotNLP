import torch.nn as nn
import torch.nn.functional as F

from model.cdssm import CDSSM


class Siames(nn.Module):

    def __init__(self, **kwargs):
        super(Siames, self).__init__()
        self.cdssm = CDSSM(**kwargs)

    def forward(self, batch_a, batch_b):
        vectors_a = self.cdssm.process_sentences(batch_a)
        vectors_b = self.cdssm.process_sentences(batch_b)

        distances = F.cosine_similarity(vectors_a, vectors_b)

        return distances
