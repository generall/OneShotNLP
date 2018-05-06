"""
This file contains functions to operate with texts
"""

import numpy as np
import zlib
from nltk import ngrams


def pad_list(batch, pad=None):
    """
    Adds padding for lists to same length.
    Used to pad sentences in batch

    :param batch: exmaple [ [1,2,3], [1,2] ]
    :param pad: value generating function to pad with
    :return:

    >>> pad_list([[1,2,3], [1,2] ], int)
    [[1, 2, 3], [1, 2, 0]]
    """

    if pad is None:
        pad = list
    batch_lengths = list(map(len, batch))
    max_len = max(batch_lengths)

    for seq, length in zip(batch, batch_lengths):
        diff = max_len - length
        for idx in range(diff):
            seq.append(pad())

    return batch


def pad_numpy(sequences, max_len=None):
    """
    Pads internal lists with zeros.
    Used to pad words in sentence.

    :param sequences - list of lists
    :param max_len:
    :return (numpy matrix with zero-padding, actual sequence lengths)

    >>> pad_numpy([[1,2,3], [1,2]])[0]
    array([[1, 2, 3],
           [1, 2, 0]])

    """
    seq_lengths = list(map(len, sequences))
    max_len = max_len or max(seq_lengths)

    seq_tensor = np.zeros((len(sequences), max_len), dtype=int)

    for idx, (seq, seqlen) in enumerate(zip(sequences, seq_lengths)):
        seq_tensor[idx, :seqlen] = np.array(seq, dtype=int)

    return seq_tensor, seq_lengths


def pad_batch(batch):
    """
    Creates fully padded batch from given

    :param batch: example: [
        [
            [1, 2, 3],
            [1],
            [1, 2]
        ],
        [
            [4, 5]
            [6]
    ]
    :return: example:
    [
        [
            [1, 2, 3],
            [1, 0, 0],
            [1, 2, 0]

        ],
        [
            [4, 5, 0],
            [6, 0, 0],
            [0, 0, 0]
        ]
    ]
    """
    max_word_len = max(map(lambda x: max(map(len, x)), batch))
    return np.stack(map(lambda x: pad_numpy(x, max_word_len)[0], pad_list(batch)))


def encode_ngram(token, dict_size):
    """
    Converts word into list of trigram hashes

    :param token:
    :param dict_size:
    :return:
    """
    token = " {} ".format(token)
    word_ngrams = []
    for ngram in ngrams(token, 3):
        crc32_hash = zlib.crc32(str(ngram).encode())
        word_ngrams.append(crc32_hash % dict_size)
    return word_ngrams


def encode_ngrams(tokens, dict_size):
    """
    Converts words into lists of trigram hashes

    :param tokens:
    :param dict_size:
    :return:
    """
    words = []
    for token in tokens:
        code = encode_ngram(token, dict_size)
        words.append(code)
    return words


def encode_texts(texts, dict_size, tokenizer=None):
    """
    Encode batch of tests into encoded representation

    :param texts:
    :param dict_size:
    :param tokenizer:
    :return:
    """
    if tokenizer is None:
        tokenizer = str.split

    return list(map(
        lambda x: encode_ngrams(x, dict_size),
        map(
            tokenizer,
            texts
        )
    ))
