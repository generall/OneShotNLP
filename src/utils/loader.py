import itertools
import os
import random
from collections import defaultdict

import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader

from config import DATA_DIR
from utils.text_tools import pad_batch, encode_texts


class MentionsLoader(DataLoader):
    """
    This is custom test loader for mentions data

    """

    test_data = os.path.join(DATA_DIR, 'data_example.tsv')

    @classmethod
    def read_lines(cls, fd):
        for line in fd:
            yield line.strip('\n').split('\t')

    def __init__(
            self,
            filename,
            read_size,
            batch_size,
            dict_size,
            tokenizer
    ):
        """

        :param filename: file to read from
        :param read_size: number of sentences to read from file per batch
        :param batch_size: size of output batch
        :param dict_size: max number of features
        :param tokenizer: function to split text into tokens
        """
        self.tokenizer = tokenizer
        self.dict_size = dict_size
        self.batch_size = batch_size
        self.read_size = read_size
        self.filename = filename
        self.mention_placeholder = "XXXXX"

    def read_batches(self):
        fd = open(self.filename, 'r', encoding='utf-8')
        reader = self.read_lines(fd)

        while True:
            batch = list(itertools.islice(reader, self.read_size))
            groups = defaultdict(list)
            for entity in batch:
                groups[entity[0]].append(entity)

            if len(groups) > 1:
                # Skip groups with only one entity
                yield groups

            if len(batch) < self.read_size:
                break

    def row_to_example(self, row):
        """
        This function converts data row to learnable example

        :param row:
        :return:
        """
        return " ".join([row[1], self.mention_placeholder, row[3]])

    def random_batch_constructor(self, groups, size):
        """
        This method generates random balanced triplets

        :return: ( [sentence], [sentence], [matches] )
        """

        sentences = []
        sentences_other = []
        match = []

        keys = list(groups.keys())

        assert len(keys) > 1

        n = 0

        while n < size:
            positive_group, negative_group = random.sample(keys, 2)

            if len(groups[positive_group]) < 2:
                continue  # Can't use this small group as a base

            base, positive = random.sample(groups[positive_group], 2)

            negative = random.choice(groups[negative_group])

            sentences.append(self.row_to_example(base))
            sentences_other.append(self.row_to_example(positive))
            match.append(1.0)  # positive pair cosine

            sentences.append(self.row_to_example(base))
            sentences_other.append(self.row_to_example(negative))
            match.append(-1.0)  # negative pair cosine

            n += 1

        return sentences, sentences_other, match

    def __iter__(self):
        """
        Iterate over data.

        :return:
        """
        for batch in self.read_batches():
            sentences_a, sentences_b, match = self.random_batch_constructor(batch, self.batch_size)

            batch_a = Variable(torch.from_numpy(pad_batch(encode_texts(sentences_a, self.dict_size, tokenizer=self.tokenizer))))
            batch_b = Variable(torch.from_numpy(pad_batch(encode_texts(sentences_b, self.dict_size, tokenizer=self.tokenizer))))
            target = Variable(torch.FloatTensor(match))

            yield batch_a, batch_b, target

    def __len__(self):
        with open(self.filename) as fd:
            num_lines = sum(1 for _ in fd)
            return int(num_lines / self.read_size)


if __name__ == '__main__':
    loader = MentionsLoader(MentionsLoader.test_data)

    batch1 = next(loader.read_batches())

    sentences, sentences_other, match = loader.random_batch_constructor(batch1, 100)

    print(sentences[3])
    print(sentences_other[3])
    print(match[3])

    print(sentences[4])
    print(sentences_other[4])
    print(match[4])
