"""
This script should perform training of the CDSSM model
"""
import nltk
import torch
from torch.autograd import Variable
import torch.optim as optim
from torchlite.torch.learner import Learner
from torchlite.torch.learner.cores import ClassifierCore
from torchlite.torch.metrics import Metric
from tqdm import tqdm

from model.siames import Siames
from utils.loader import MentionsLoader


def tokenizer(text, alpha_only=True):  # create a tokenizer function
    return [tok for tok in nltk.word_tokenize(text) if (not alpha_only or tok.isalpha())]


def loss_foo(distances, target, alpha=0.4):
    """

    :param alpha: minimal distance
    :param distances: 1d Tensor shape: (num_examples, )
    :param target: 1d Tensor shape: (num_examples, )
    """

    diff = torch.abs(distances - target)
    return torch.sum(diff[diff > alpha])


class DistAccuracy(Metric):

    def __init__(self, alpha=0.4):
        self.alpha = alpha

    @property
    def get_name(self):
        return "dist_accuracy"

    def __call__(self, y_pred, y_true):
        diff = torch.abs(y_pred - y_true)
        positive = torch.sum((diff > self.alpha).int()).data.item()
        total = y_true.shape[0]
        return positive / total


epoch_max = 10
read_size = 500
batch_size = 10
dict_size = 20000

use_cuda = False

train_loader = MentionsLoader(MentionsLoader.test_data, read_size=read_size, batch_size=batch_size, dict_size=dict_size,
                              tokenizer=tokenizer)

test_loader = MentionsLoader(MentionsLoader.test_data, read_size=read_size, batch_size=batch_size, dict_size=dict_size,
                             tokenizer=tokenizer)

model = Siames()

optimizer = optim.RMSprop(model.parameters(), lr=1e-3)

loss = loss_foo

learner = Learner(ClassifierCore(model, optimizer, loss), use_cuda=use_cuda)

metrics = [DistAccuracy()]

learner.train(epoch_max, metrics, train_loader, test_loader, callbacks=None)

