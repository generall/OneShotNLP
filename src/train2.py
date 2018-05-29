"""
This script should perform training of the CDSSM model
"""
import argparse

import os
import nltk
import torch
import torch.optim as optim
from torchlite.torch.learner import Learner
from torchlite.torch.learner.cores import ClassifierCore
from torchlite.torch.metrics import Metric
from torchlite.torch.train_callbacks import TensorboardVisualizerCallback, ModelSaverCallback

from config import TB_DIR, MODELS_DIR
from model.siames import Siames
from utils.loader import MentionsLoader


def tokenizer(text, alpha_only=True):  # create a tokenizer function
    return [tok for tok in nltk.word_tokenize(text) if (not alpha_only or tok.isalpha())]


def loss_foo(distances, target):
    """

    :param alpha: minimal distance
    :param distances: 1d Tensor shape: (num_examples, )
    :param target: 1d Tensor shape: (num_examples, )
    """

    diff = torch.abs(distances - target)
    return torch.sum(diff)


class DistAccuracy(Metric):

    def __init__(self, alpha=0.4):
        self.alpha = alpha

    @property
    def get_name(self):
        return "dist_accuracy"

    def __call__(self, y_pred, y_true):
        """

        :param y_pred: Distance between objects.
        :param y_true: ???
        :return:
        """

        positive = torch.sum((y_pred * y_true > 0).int()).data.item()
        total = y_true.shape[0]
        return positive / total


parser = argparse.ArgumentParser(description='Train One Shot CDSSM')

parser.add_argument('--train-data', dest='train_data', help='path to train data', default=MentionsLoader.test_data)
parser.add_argument('--valid-data', dest='valid_data', help='path to valid data', default=MentionsLoader.test_data)

parser.add_argument('--restore-model', dest='restore_model', help='path to saved model', default=os.path.join(MODELS_DIR, 'Siames_epoch-500.pth'))

parser.add_argument('--epoch', type=int, default=100)
parser.add_argument('--save-every', type=int, default=10)
parser.add_argument('--read-size', type=int, default=500)
parser.add_argument('--batch-size', type=int, default=100)
parser.add_argument('--dict-size', type=int, default=50000)
parser.add_argument('--cuda', type=bool, default=False)

args = parser.parse_args()

train_loader = MentionsLoader(
    args.train_data,
    read_size=args.read_size,
    batch_size=args.batch_size,
    dict_size=args.dict_size,
    tokenizer=tokenizer
)

test_loader = MentionsLoader(
    args.valid_data,
    read_size=args.read_size,
    batch_size=args.batch_size,
    dict_size=args.dict_size,
    tokenizer=tokenizer
)

loss = loss_foo

model = Siames(
    word_emb_sizes=[1000, 500],
    conv_sizes=[400, 300],
    out_size=[256, 128],
    embedding_size=args.dict_size
)

if args.restore_model:
    ModelSaverCallback.restore_model_from_file(model, args.restore_model, load_with_cpu=(not args.cuda))

optimizer = optim.Adam(model.parameters(), lr=1e-3)


metrics = [DistAccuracy()]
callbacks = [
    TensorboardVisualizerCallback(TB_DIR),
    ModelSaverCallback(MODELS_DIR, epochs=args.epoch, every_n_epoch=args.save_every)
]

learner = Learner(ClassifierCore(model, optimizer, loss), use_cuda=args.cuda)
learner.train(args.epoch, metrics, train_loader, test_loader, callbacks=callbacks)
