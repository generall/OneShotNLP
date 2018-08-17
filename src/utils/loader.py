import gzip
import itertools
import random
import re
from collections import defaultdict
from zipfile import ZipFile

import torch
from torch.utils.data import DataLoader

from utils.text_tools import pad_batch, encode_texts, pad_list

from multiprocessing import Pool, Semaphore

import numpy as np


class BatchReader:

    @classmethod
    def get_file_iterator(cls, filename):
        if re.match(r'.*\.gz$', filename):
            fd = gzip.open(filename, 'rt', encoding='utf-8')
        elif re.match(r'.*\.zip$', filename):
            arch = ZipFile(filename)
            zipname = arch.namelist()[0]
            fd = map(lambda x: x.decode('utf-8'), arch.open(zipname))
        else:
            fd = open(filename, 'r', encoding='utf-8')

        return fd

    @classmethod
    def read_lines(cls, fd):
        for line in fd:
            yield line.strip('\n').split('\t')

    def read_batches(self, semaphore=None):
        raise NotImplementedError()

    def __len__(self):
        raise NotImplementedError()


class WikiLinksReader(BatchReader):

    def __init__(self, filename, read_size, batch_size):
        self.batch_size = batch_size
        self.read_size = read_size
        self.filename = filename
        self.mention_placeholder = "XXXXX"
        self.file_len = None

    def row_to_example(self, row):
        """
        This function converts data row to learnable example

        :param row:
        :return:
        """
        length_limit = 150  # pair of tweets
        return " ".join([row[1][-length_limit:], self.mention_placeholder, row[3][:length_limit]]).strip()

    def random_batch_constructor(self, groups, size=None):
        """
        This method generates random balanced triplets

        :return: ( [sentence], [sentence], [matches] )
        """

        size = size or self.batch_size

        sentences = []
        sentences_other = []
        match = []

        keys = list(groups.keys())

        assert len(keys) > 1

        n = 0

        tries = 10000

        while n < size:
            positive_group, negative_group = random.sample(keys, 2)

            if len(groups[positive_group]) < 2:
                tries -= 1
                if tries < 0:
                    raise RuntimeWarning("Bad bath, can't find any group larges than 1")
                continue  # Can't use this small group as a base

            tries = 10000

            base, positive = random.sample(groups[positive_group], 2)

            negative = random.choice(groups[negative_group])

            if random.random() > 0.5:
                sentences.append(self.row_to_example(base))
                sentences_other.append(self.row_to_example(positive))
            else:
                sentences.append(self.row_to_example(positive))
                sentences_other.append(self.row_to_example(base))

            if random.random() > 0.5:
                sentences.append(self.row_to_example(base))
                sentences_other.append(self.row_to_example(negative))
            else:
                sentences.append(self.row_to_example(negative))
                sentences_other.append(self.row_to_example(base))

            match.append(1)  # positive pair cosine
            match.append(0)  # negative pair cosine

            n += 1

        return sentences, sentences_other, match

    def read_batches(self, semaphore=None):
        fd = self.get_file_iterator(self.filename)
        reader = self.read_lines(fd)

        while True:
            if semaphore is not None:
                semaphore.acquire()

            batch = list(itertools.islice(reader, self.read_size))
            groups = defaultdict(list)
            for entity in batch:
                groups[entity[0]].append(entity)

            if len(groups) > 1:
                # Skip groups with only one entity
                yield self.random_batch_constructor(groups)

            if len(batch) < self.read_size:
                break

    def __len__(self):
        if self.file_len is not None:
            return self.file_len
        else:
            fd = self.get_file_iterator(self.filename)
            num_lines = sum(1 for _ in fd)
            self.file_len = int(num_lines / self.read_size)
            return self.file_len


class SimpleReader(BatchReader):

    def __init__(self, filename, batch_size):
        self.batch_size = batch_size
        self.filename = filename
        self.file_len = None

    def read_batches(self, semaphore=None):
        fd = self.get_file_iterator(self.filename)
        reader = self.read_lines(fd)

        batch_sent_a = []
        batch_sent_b = []
        batch_match = []
        for row in reader:
            sent_a, sent_b, match = row
            match = int(match)

            batch_sent_a.append(sent_a)
            batch_sent_b.append(sent_b)
            batch_match.append(match)

            if len(batch_sent_a) > self.batch_size:
                yield batch_sent_a, batch_sent_b, batch_match

                batch_sent_a = []
                batch_sent_b = []
                batch_match = []

        if len(batch_sent_a) > 0:
            yield batch_sent_a, batch_sent_b, batch_match

    def __len__(self):
        if self.file_len is not None:
            return self.file_len
        else:
            fd = self.get_file_iterator(self.filename)
            num_lines = sum(1 for _ in fd)
            self.file_len = int(num_lines / self.batch_size)
            return self.file_len


class MentionsLoader(DataLoader):
    """
    This is custom test loader for mentions data

    """

    def __init__(
            self,
            batch_reader,
            tokenizer,
            parallel=0,
    ):
        """

        :param batch_reader: object to read raw batches from file
        :param tokenizer: function to split text into tokens
        :param parallel: number of data preparation processes
        """
        self.batch_reader = batch_reader
        self.parallel = parallel
        self.tokenizer = tokenizer

    @property
    def batch_size(self):
        return self.batch_reader.batch_size

    def construct_rels(self, sentences_a, sentences_b, match):
        raise NotImplementedError()

    def full_construct(self, batch):
        return self.construct_rels(*batch)

    @classmethod
    def to_torch(cls, batch: tuple):
        return tuple(map(torch.from_numpy, batch))

    def iter_batches(self):
        if self.parallel > 0:
            with Pool(self.parallel) as pool:
                semaphore = Semaphore(self.parallel * 10)
                for batch in pool.imap(self.full_construct, self.batch_reader.read_batches(semaphore)):
                    yield batch
                    semaphore.release()
        else:
            for batch in self.batch_reader.read_batches():
                yield self.full_construct(batch)

    def __iter__(self):
        """
        Iterate over data.

        :return:
        """
        yield from map(MentionsLoader.to_torch, self.iter_batches())

    def __len__(self):
        return len(self.batch_reader)


class HashingMentionLoader(MentionsLoader):

    def __init__(self, batch_reader, tokenizer, ngrams_flag, dict_size, **kwargs):
        super().__init__(batch_reader, tokenizer, **kwargs)
        self.ngrams_flag = ngrams_flag
        self.dict_size = dict_size

    def construct_rels(self, sentences_a, sentences_b, match):
        batch_a = torch.from_numpy(pad_batch(
            encode_texts(sentences_a, self.dict_size, tokenizer=self.tokenizer, ngrams=self.ngrams_flag)))
        batch_b = torch.from_numpy(pad_batch(
            encode_texts(sentences_b, self.dict_size, tokenizer=self.tokenizer, ngrams=self.ngrams_flag)))
        target = torch.LongTensor(match)
        return batch_a, batch_b, target


class WordMentionLoader(MentionsLoader):
    """
    Mention loader which return raw words instead of hashed ngrams
    """

    def construct_rels(self, sentences_a, sentences_b, match):
        batch_a = pad_list(list(map(
            self.tokenizer,
            sentences_a
        )), pad=lambda: " ")

        batch_b = pad_list(list(map(
            self.tokenizer,
            sentences_b
        )), pad=lambda: " ")

        target = torch.LongTensor(match)
        return batch_a, batch_b, target


class EmbeddingMentionLoader(MentionsLoader):
    vectorizer = None

    def __init__(self, batch_reader, tokenizer, vectorizer, **kwargs):
        super().__init__(batch_reader, tokenizer, **kwargs)
        self.__class__.vectorizer = vectorizer

    def construct_rels(self, sentences_a, sentences_b, match):
        batch_a = self.__class__.vectorizer(pad_list(list(map(
            self.tokenizer,
            sentences_a
        )), pad=lambda: " ")).numpy()

        # Convert to numpy to force serialization in multiprocessing
        # https://github.com/pytorch/pytorch/issues/973

        batch_b = self.vectorizer(pad_list(list(map(
            self.tokenizer,
            sentences_b
        )), pad=lambda: " ")).numpy()

        target = self.target_prep(np.array(match))
        return batch_a, batch_b, target

    def target_prep(self, target):
        return target.astype(np.float32)


class EmbeddingMentionLoaderCos(EmbeddingMentionLoader):
    def target_prep(self, target):
        return target.astype(np.float32) * 2 - 1
