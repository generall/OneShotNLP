import os
import time
from unittest import TestCase

import nltk

from config import DATA_DIR
from utils.loader import MentionsLoader


def tokenizer(text, alpha_only=True):  # create a tokenizer function
    return [tok for tok in nltk.word_tokenize(text) if (not alpha_only or tok.isalpha())]


class TestMentionsLoader(TestCase):
    def test_read_batches(self):
        loader = MentionsLoader(
            os.path.join(DATA_DIR, 'debug_data', 'syntetic_6_train.tsv'),
            read_size=250,
            batch_size=1000,
            dict_size=1000,
            tokenizer=tokenizer,
            ngrams_flag=False,
            parallel=2
        )
        load_iter = iter(loader)
        for i in range(10):
            ts = time.time()
            a1, b1, match = next(load_iter)
            te = time.time()
            print(a1.shape, (te - ts) * 1000)
