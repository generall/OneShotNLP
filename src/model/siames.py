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

    def forward(self, batch_a, batch_b):
        vectors_a = self.cdssm.process_sentences(batch_a)
        vectors_b = self.cdssm.process_sentences(batch_b)

        out = self.sm(self.combination_layer(vectors_a * vectors_b)).squeeze()

        return out


class Matcher(nn.Module):

    def __init__(self, out_size, **kwargs):
        super(Matcher, self).__init__()
        self.out_size = out_size
        self.cdssm_a = CDSSM(out_size=out_size, **kwargs)
        self.cdssm_b = CDSSM(out_size=out_size, **kwargs)

        self.combination_layer = nn.Linear(self.out_size[-1], 1)
        self.sm = nn.Sigmoid()

    def forward(self, batch_a, batch_b):
        vectors_a = self.cdssm_a.process_sentences(batch_a)
        vectors_b = self.cdssm_b.process_sentences(batch_b)

        out = self.sm(self.combination_layer(vectors_a * vectors_b)).squeeze()

        return out


class SiamesCos(nn.Module):

    def __init__(self, out_size, **kwargs):
        super(SiamesCos, self).__init__()
        self.out_size = out_size
        self.cdssm = CDSSM(out_size=out_size, **kwargs)
        self.cos = torch.nn.CosineSimilarity()

    def forward(self, batch_a, batch_b):
        vectors_a = self.cdssm.process_sentences(batch_a)
        vectors_b = self.cdssm.process_sentences(batch_b)

        return vectors_a, vectors_b
