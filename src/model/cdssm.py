import itertools
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


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


def pairwise(iterable):
    """
    s -> (s0,s1), (s1,s2), (s2, s3), ...

    :param iterable:
    :return:
    """
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)


class CDSSM(nn.Module):

    def __init__(
            self,
            word_emb_sizes=None,
            conv_sizes=None,
            out_size=None,
            window=3,
            embedding_size=20000,
    ):
        super(CDSSM, self).__init__()

        activation = nn.Tanh

        if out_size is None:
            out_size = [128]

        if conv_sizes is None:
            conv_sizes = [300]

        if word_emb_sizes is None:
            word_emb_sizes = [500]

        self.embedding_size = embedding_size
        self.window = window
        self.out_size = out_size
        self.conv_sizes = conv_sizes
        self.word_emb_sizes = word_emb_sizes

        sparse_linear = SparseLinear(dict_size=self.embedding_size, out_features=self.word_emb_sizes[0])

        input_to_word_vect = [sparse_linear]

        for from_size, to_size in pairwise(self.word_emb_sizes):
            input_to_word_vect += [
                nn.Linear(from_size, to_size),
                activation()
            ]

        self.input_to_vect = nn.Sequential(*input_to_word_vect)

        convolutions = [
            torch.nn.Conv1d(self.word_emb_sizes[-1], self.conv_sizes[0], self.window),
            activation()
        ]

        for from_size, to_size in pairwise(self.conv_sizes):
            convolutions += [
                torch.nn.Conv1d(from_size, to_size, self.window),
                activation()
            ]

        self.convolution = nn.Sequential(*convolutions)

        feed_forward = [
            nn.Linear(in_features=self.conv_sizes[-1], out_features=self.out_size[0]),
            activation()
        ]

        for from_size, to_size in pairwise(self.out_size):
            convolutions += [
                torch.nn.Linear(from_size, to_size),
                activation()
            ]

        self.feed_forward = nn.Sequential(*feed_forward)

    def weight_init(self, init_foo):

        def init_weights(m):
            if isinstance(m, (nn.Linear, nn.Conv1d)):
                init_foo(m.weight)
                # nn.init.xavier_uniform_(m.bias, gain=nn.init.calculate_gain('tanh'))

        self.input_to_vect.apply(init_weights)
        self.convolution.apply(init_weights)
        self.feed_forward.apply(init_weights)

    def process_sentences(self, sentences):
        """
        :param sentences Tensor (batch_size, sentence_length, word_depth)
        """

        # Compress sparse ngram representation into dense vectors
        sentences = self.input_to_vect(sentences)

        # Prepare for convolution and apply it.
        # Combine 3-word window into single vector
        sentences = sentences.transpose(1, 2)

        conv_embedding = self.convolution(sentences)

        # Apply max-pooling to compress variable-length sequence of 3-word vectors into single document vector
        convolutions_size = conv_embedding.size()[2]
        max_pooling = F.max_pool1d(conv_embedding, kernel_size=convolutions_size).view(-1, self.conv_sizes[-1])

        # Compress pooled representation even more
        res = self.feed_forward(max_pooling)

        return res
