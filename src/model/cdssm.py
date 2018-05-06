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


class CDSSM(nn.Module):

    def __init__(self,
                 conv_input_size=500,
                 conv_out_size=300,
                 out_size=128,
                 window=3,
                 embedding_size=20000,
                 is_cuda=False):
        super(CDSSM, self).__init__()

        self.embedding_size = embedding_size
        self.window = window
        self.out_size = out_size
        self.conv_out_size = conv_out_size
        self.conv_input_size = conv_input_size
        self.is_cuda = is_cuda

        self.sparse_linear = SparseLinear(dict_size=self.embedding_size, out_features=self.conv_input_size)
        self.conv_nn = torch.nn.Conv1d(self.conv_input_size, self.conv_out_size, self.window)
        self.feed_forvard = nn.Linear(in_features=self.conv_out_size, out_features=self.out_size)

        if self.is_cuda:
            self.cuda()

    def process_sentence(self, sentences):
        """
        :param sentences Tensor (batch_size, sentence_length, word_depth)
        """

        # Compress sparse ngram representation into dense vectors
        sentences = F.relu(self.sparse_linear(sentences))

        # Prepare for convolution and apply it.
        # Combine 3-word window into single vector
        sentences = sentences.transpose(1, 2)

        conv_embedding = F.relu(self.conv_nn(sentences))

        # Apply max-pooling to compress variable-length sequence of 3-word vectors into single document vector
        convolutions_size = conv_embedding.size()[2]
        max_pooling = F.max_pool1d(conv_embedding, kernel_size=convolutions_size).view(-1, self.conv_out_size)

        # Compress pooled representation even more
        res = F.relu(self.feed_forvard(max_pooling))

        return res
