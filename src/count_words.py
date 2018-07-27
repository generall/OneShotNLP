import sys

import nltk


def tokenizer(text, alpha_only=True):  # create a tokenizer function
    words = [tok for tok in nltk.word_tokenize(text) if (not alpha_only or tok.isalpha())]
    return words


with open(sys.argv[1]) as fd:
    max_words_count = 0
    for idx, line in enumerate(fd):
        _, a, _, b = line.strip('\n').split('\t')

        text = a + ' ' + b

        tokens = tokenizer(text)

        wc = len(tokens)

        if wc > max_words_count:
            max_words_count = wc
            print("Max count: ", wc, '#', idx,  a[:30])
