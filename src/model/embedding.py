import itertools

import fastText
import torch
from torch.nn import EmbeddingBag
import torch.nn as nn
import numpy as np

from utils.disc_vectors import DiscVectors


class FastTextEmbeddingBag(EmbeddingBag):
    def __init__(self, model_path, learn_emb=False):
        self.model = fastText.load_model(model_path)
        input_matrix = self.model.get_input_matrix()
        input_matrix_shape = input_matrix.shape
        super().__init__(input_matrix_shape[0], input_matrix_shape[1])
        self.weight.data.copy_(torch.FloatTensor(input_matrix))
        self.weight.requires_grad = learn_emb

    def forward(self, words, offsets=None):
        word_subinds = np.empty([0], dtype=np.int64)
        word_offsets = [0]
        for word in words:
            _, subinds = self.model.get_subwords(word)
            word_subinds = np.concatenate((word_subinds, subinds))
            word_offsets.append(word_offsets[-1] + len(subinds))
        word_offsets = word_offsets[:-1]
        ind = torch.LongTensor(word_subinds)
        offsets = torch.LongTensor(word_offsets)

        return super().forward(ind, offsets)


class EmbeddingVectorizer(nn.Module):
    def __init__(self, embedding):
        super(EmbeddingVectorizer, self).__init__()
        self.embedding = embedding

    def forward(self, batch):
        batch_size = len(batch)
        sent_len = len(batch[0])
        flatten = list(itertools.chain(*batch))

        return self.embedding(flatten).view(batch_size, sent_len, -1)


class ModelVectorizer:

    def __init__(self, model_path):
        self.model = fastText.load_model(model_path)

    def convert(self, batch):
        mtx_batch = []
        for sent in batch:
            mtx_sent = []
            for token in sent:
                mtx_sent.append(self.model.get_word_vector(token))
            mtx_sent = np.stack(mtx_sent)
            mtx_batch.append(mtx_sent)
        mtx_batch = np.stack(mtx_batch)

        return torch.from_numpy(mtx_batch).float()


class OnDiskVectorizer(ModelVectorizer):

    def __init__(self, mtx_path, meta_path):
        self.model = DiscVectors(mtx_path, meta_path)