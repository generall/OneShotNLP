import itertools
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from model.sparse_linear import SparseLinear, pairwise


class MatchMatrix(nn.Module):

    def __init__(
            self,
            input_size,
            matrix_depth
    ):
        super(MatchMatrix, self).__init__()
        activation = nn.Tanh

        self.matrix_depth = matrix_depth

        interaction = [
            nn.Linear(input_size * 2, matrix_depth[0]),
            activation()
        ]

        for from_size, to_size in pairwise(self.matrix_depth):
            interaction += [
                nn.Linear(from_size, to_size),
                activation()
            ]

        self.interaction = nn.Sequential(*interaction)

    def forward(self, sent_a, sent_b):
        a_size = sent_a.shape[1]
        b_size = sent_a.shape[1]

        a = torch.stack([sent_a] * b_size, dim=2)
        b = torch.stack([sent_b] * a_size, dim=1)

        matrix = torch.cat([a, b], dim=3)

        return self.interaction(matrix)


class ARC2(nn.Module):

    def __init__(
            self,
            word_emb_sizes,
            matrix_depth,
            conv_depth,
            out_size,
            embedding_size,
            window
    ):
        super(ARC2, self).__init__()
        self.window = window
        self.embedding_size = embedding_size
        self.out_size = out_size
        self.conv_depth = conv_depth
        self.matrix_depth = matrix_depth
        self.word_emb_sizes = word_emb_sizes
        activation = nn.Tanh

        sparse_linear = SparseLinear(dict_size=self.embedding_size, out_features=self.word_emb_sizes[0])

        input_to_word_vect = [sparse_linear]

        for from_size, to_size in pairwise(self.word_emb_sizes):
            input_to_word_vect += [
                nn.Linear(from_size, to_size),
                activation()
            ]

        self.input_to_vect = nn.Sequential(*input_to_word_vect)

        self.match_layer = MatchMatrix(self.word_emb_sizes[-1], self.matrix_depth)

        convolutions = [
            torch.nn.Conv2d(self.matrix_depth[-1], self.conv_depth[0], self.window),
            activation()
        ]

        for from_size, to_size in pairwise(self.conv_depth):
            convolutions += [
                torch.nn.Conv2d(from_size, to_size, self.window),
                activation()
            ]

        self.convolution = nn.Sequential(*convolutions)

        feed_forward = [
            nn.Linear(in_features=self.conv_depth[-1], out_features=self.out_size[0])
        ]

        for from_size, to_size in pairwise(self.out_size):
            feed_forward += [
                activation(),
                torch.nn.Linear(from_size, to_size)
            ]

        feed_forward.append(nn.Softmax(dim=1))

        self.feed_forward = nn.Sequential(*feed_forward)

    def forward(self, sent_a, sent_b):

        sent_embedding_a = self.input_to_vect(sent_a)
        sent_embedding_b = self.input_to_vect(sent_b)

        match_matrix = self.match_layer(sent_embedding_a, sent_embedding_b)

        match_matrix = match_matrix.transpose(3, 1)  # Output: (N, Channels, Height, Width)

        conv_matrix = self.convolution(match_matrix)

        _, _, h, w = conv_matrix.shape

        max_pooling = F.max_pool2d(conv_matrix, kernel_size=(h, w)).view(-1, self.conv_depth[-1])

        res = self.feed_forward(max_pooling)


        return res







