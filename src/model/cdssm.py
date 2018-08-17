import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.sparse_linear import SparseLinear, pairwise


class CDSSM(nn.Module):

    def __init__(
            self,
            word_emb_sizes,
            conv_sizes,
            out_size,
            dropout=0.2,
            window=3,
    ):
        super(CDSSM, self).__init__()

        activation = nn.LeakyReLU

        final_activation = nn.Tanh

        self.dropout = dropout
        self.window = window
        self.out_size = out_size
        self.conv_sizes = conv_sizes
        self.word_emb_sizes = word_emb_sizes

        convolutions = [
            torch.nn.Conv1d(self.word_emb_sizes, self.conv_sizes[0], self.window),
            activation(),
            torch.nn.Dropout(self.dropout)
        ]

        for from_size, to_size in pairwise(self.conv_sizes):
            convolutions += [
                torch.nn.MaxPool1d(2, padding=1),
                torch.nn.Conv1d(from_size, to_size, self.window),
                activation(),
                torch.nn.Dropout(self.dropout)
            ]

        self.convolution = nn.Sequential(*convolutions)

        feed_forward = [
            nn.Linear(in_features=self.conv_sizes[-1] * 2, out_features=self.out_size[0]),
            activation(),
            torch.nn.Dropout(self.dropout)
        ]

        for from_size, to_size in pairwise(self.out_size):
            feed_forward += [
                torch.nn.Linear(from_size, to_size),
                final_activation(),
                torch.nn.Dropout(self.dropout)
            ]

        self.feed_forward = nn.Sequential(*feed_forward)

    def process_sentences(self, sentences):
        """
        :param sentences Tensor (batch_size, sentence_length, word_depth)
        """

        # Prepare for convolution and apply it.
        # Combine 3-word window into single vector
        sentences = sentences.transpose(1, 2)

        conv_embedding = self.convolution(sentences)

        # Apply max-pooling to compress variable-length sequence of 3-word vectors into single document vector
        convolutions_size = conv_embedding.size()[-1]

        max_pooling = F.max_pool1d(conv_embedding, kernel_size=convolutions_size).squeeze()
        avg_pooling = F.avg_pool1d(conv_embedding, kernel_size=convolutions_size).squeeze()

        pooling = torch.cat([max_pooling, avg_pooling], dim=1)

        # Compress pooled representation even more
        res = self.feed_forward(pooling)

        return res
