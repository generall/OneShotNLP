import json
import zlib

import numpy as np


class DiscVectors:
    """
    This class provides an interface to a word vectors without reading it in memory.
    """

    def read_meta(self, meta_path):
        with open(meta_path) as fd:
            meta = json.load(fd)

        self.dim = meta['dim']
        self.vocab = meta['vocab']
        self.length = meta['length']

    def __init__(self, matrix_path, meta_path, default_word=" "):
        self.default_word = default_word
        self.vocab = {}
        self.dim = 0
        self.length = 0
        self.read_meta(meta_path)
        self.matrix = np.memmap(matrix_path, dtype='float32', mode='r', shape=(self.length, self.dim))

    def get_word_vector(self, word):
        word_id = self.vocab.get(word) or self.vocab.get(word.lower())
        if word_id is None:
            seed = zlib.crc32(word.encode())
            state = np.random.get_state()
            np.random.seed(seed)
            vect = np.random.normal(size=self.dim)
            np.random.set_state(state)
            return vect

        return self.matrix[word_id]

    @classmethod
    def convert_fastText(cls, ft_path, output_vectors_path, output_meta_path, default_word=" ", additional_vocab=None):
        """
        Convert fastText model into accessible from disk representation

        :param output_meta_path: Path used to save metadata (vocab + dimensions count)
        :param output_vectors_path: Path used to save vectors
        :param ft_path: Path to fastText .bin model
        :param default_word: word to use if OOV
        :param additional_vocab: generate additional word vectors and add it to vocab
        :return:
        """
        if additional_vocab is None:
            additional_vocab = []
        import tqdm
        import fastText
        model = fastText.load_model(ft_path)

        vocab = model.f.getVocab()[0]
        vocab.append(default_word)

        vocab = vocab + additional_vocab

        fp = np.memmap(output_vectors_path, dtype='float32', mode='w+', shape=(len(vocab), model.get_dimension()))

        idx_vocab = {}

        for idx, word in tqdm.tqdm(enumerate(vocab)):
            vect = model.get_word_vector(word)
            fp[idx] = vect
            idx_vocab[word] = idx

        del fp

        with open(output_meta_path, 'w') as fd:
            json.dump({
                'length': len(vocab),
                'vocab': idx_vocab,
                'dim': model.get_dimension()
            }, fd)

    @classmethod
    def convert_gensim_fasttext(cls, ft_path, output_vectors_path, output_meta_path, default_word=" ",
                                additional_vocab=None):
        """
        Convert fastText model into accessible from disk representation

        :param output_meta_path: Path used to save metadata (vocab + dimensions count)
        :param output_vectors_path: Path used to save vectors
        :param ft_path: Path to fastText .bin model
        :param default_word: word to use if OOV
        :param additional_vocab: generate additional word vectors and add it to vocab
        :return:
        """
        if additional_vocab is None:
            additional_vocab = []
        import tqdm
        import gensim
        model = gensim.models.FastText.load(ft_path)

        vocab = list(model.wv.vocab.keys())
        vocab.append(default_word)

        vocab = vocab + additional_vocab

        fp = np.memmap(output_vectors_path, dtype='float32', mode='w+', shape=(len(vocab), model.wv.vector_size))

        idx_vocab = {}

        for idx, word in tqdm.tqdm(enumerate(vocab)):
            try:
                vect = model.wv.get_vector(word)
                fp[idx] = vect
                idx_vocab[word] = idx
            except KeyError:
                pass

        del fp

        with open(output_meta_path, 'w') as fd:
            json.dump({
                'length': len(vocab),
                'vocab': idx_vocab,
                'dim': model.wv.vector_size
            }, fd)
