import argparse
import random

import nltk

from model.embedding import EmbeddingVectorizer, FastTextEmbeddingBag
from utils.loader import MentionsLoader, WordMentionLoader, EmbeddingMentionLoader

parser = argparse.ArgumentParser(description='Prepare cache for train data')


def tokenizer(text, alpha_only=True):  # create a tokenizer function
    return [tok for tok in nltk.word_tokenize(text) if (not alpha_only or tok.isalpha())]


parser.add_argument('--train-data', dest='train_data', help='path to train data', default=MentionsLoader.debug_train)
parser.add_argument('--valid-data', dest='valid_data', help='path to valid data', default=MentionsLoader.debug_valid)

parser.add_argument('--read-size', type=int, default=250)
parser.add_argument('--batch-size', type=int, default=1000)
parser.add_argument('--dict-size', type=int, default=50000)
parser.add_argument('--ngram', type=bool, default=False)
parser.add_argument('--parallel', type=int, default=0)
parser.add_argument('--cycles', type=int, default=1)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--emb-path', type=str, default=None)

args = parser.parse_args()
random.seed(args.seed)

vectorizer = EmbeddingVectorizer(FastTextEmbeddingBag(model_path=args.emb_path))

train_loader = EmbeddingMentionLoader(
    args.train_data,
    read_size=args.read_size,
    batch_size=args.batch_size,
    dict_size=args.dict_size,
    tokenizer=tokenizer,
    ngrams_flag=args.ngram,
    parallel=args.parallel,
    cycles=args.cycles,
    vectorizer=vectorizer
)

test_loader = EmbeddingMentionLoader(
    args.valid_data,
    read_size=args.read_size,
    batch_size=args.batch_size,
    dict_size=args.dict_size,
    tokenizer=tokenizer,
    ngrams_flag=args.ngram,
    parallel=args.parallel,
    cycles=args.cycles,
    vectorizer=vectorizer
)


train_loader.cache_pickle(args.train_data + '.pkl')
test_loader.cache_pickle(args.valid_data + '.pkl')
