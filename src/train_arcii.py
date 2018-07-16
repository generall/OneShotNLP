"""
This script should perform training of the CDSSM model
"""
import argparse
import datetime

import os
import random

import torch

import nltk
import torch.optim as optim
from torchlite.torch.learner import Learner
from torchlite.torch.learner.cores import ClassifierCore
from torchlite.torch.train_callbacks import TensorboardVisualizerCallback, ModelSaverCallback, ReduceLROnPlateau

from config import TB_DIR, MODELS_DIR
from model.arc2 import ARC2, PreConv, SumVectorizer, EmbeddingVectorizer, FastTextEmbeddingBag
from model.loss import AccuracyMetric, CrossEntropyLoss
from utils.loader import MentionsLoader, WordMentionLoader
from utils.loggers import ModelParamsLogger


def tokenizer(text, alpha_only=True):  # create a tokenizer function
    return [tok for tok in nltk.word_tokenize(text) if (not alpha_only or tok.isalpha())]


parser = argparse.ArgumentParser(description='Train One Shot CDSSM')

parser.add_argument('--train-data', dest='train_data', help='path to train data', default=MentionsLoader.debug_train)
parser.add_argument('--valid-data', dest='valid_data', help='path to valid data', default=MentionsLoader.debug_valid)

parser.add_argument('--restore-model', dest='restore_model',
                    help='path to saved model')  # default=os.path.join(MODELS_DIR, 'Siames_epoch-150.pth'))

parser.add_argument('--epoch', type=int, default=100)
parser.add_argument('--save-every', type=int, default=10)
parser.add_argument('--read-size', type=int, default=250)
parser.add_argument('--batch-size', type=int, default=1000)
parser.add_argument('--dict-size', type=int, default=50000)

parser.add_argument('--cuda', type=bool, default=False)
parser.add_argument('--ngram', type=bool, default=False)

parser.add_argument('--parallel', type=int, default=0)

parser.add_argument('--patience', type=int, default=10)

parser.add_argument('--run', default='none', help='name of current run for tensorboard')

parser.add_argument('--lr', type=float, default=1e-2)

parser.add_argument('--weight_decay', type=float, default=0)

parser.add_argument('--netsize', type=int, default=10)

parser.add_argument('--emb-size', type=int, default=50)

parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--torch-seed', type=int, default=42)

parser.add_argument('--cycles', type=int, default=1)

parser.add_argument('--dropout', type=float, default=0.0)

parser.add_argument('--preconv', type=bool, default=False, help="Use conv1d layer before interaction matrix")

parser.add_argument('--emb-path', type=str, default=None)

args = parser.parse_args()

random.seed(args.seed)
torch.manual_seed(args.torch_seed)

train_loader = WordMentionLoader(
    args.train_data,
    read_size=args.read_size,
    batch_size=args.batch_size,
    dict_size=args.dict_size,
    tokenizer=tokenizer,
    ngrams_flag=args.ngram,
    parallel=args.parallel,
    cycles=args.cycles,
)

test_loader = WordMentionLoader(
    args.valid_data,
    read_size=args.read_size,
    batch_size=args.batch_size,
    dict_size=args.dict_size,
    tokenizer=tokenizer,
    ngrams_flag=args.ngram,
    parallel=args.parallel,
    cycles=args.cycles,
)

loss = CrossEntropyLoss()

preconv_size = [args.emb_size] if args.preconv else None

vectorizer = EmbeddingVectorizer(FastTextEmbeddingBag(model_path=args.emb_path))

preconv = PreConv(
    word_emb_sizes=[args.emb_size],
    sent_conv_size=preconv_size,
    dropout=args.dropout,
    window=2
)

model = ARC2(
    vectorizer=vectorizer,
    preconv=preconv,
    matrix_depth=[args.netsize],
    conv_depth=[args.netsize],
    out_size=[args.netsize],
    window=2,
    dropout=args.dropout
)

gain = torch.nn.init.calculate_gain('leaky_relu')
model.weight_init(lambda x: torch.nn.init.xavier_normal_(x, gain=gain))

if args.restore_model:
    ModelSaverCallback.restore_model_from_file(model, args.restore_model, load_with_cpu=(not args.cuda))

optimizer = optim.Adam(filter(lambda x: x.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.weight_decay)

run_name = args.run + '-' + datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")

tb_dir = os.path.join(TB_DIR, run_name)
if not os.path.exists(tb_dir):
    os.mkdir(tb_dir)

metrics = [
    AccuracyMetric(),
]


class MyReduceLROnPlateau(ReduceLROnPlateau):
    def on_epoch_end(self, epoch, logs=None):
        step = logs["step"]
        if step == 'validation':
            batch_logs = logs.get('batch_logs', {})
            epoch_loss = batch_logs.get('loss')
            if epoch_loss is not None:
                print('reduce lr num_bad_epochs: ', self.lr_sch.num_bad_epochs)
                self.lr_sch.step(epoch_loss, epoch)


callbacks = [
    ModelParamsLogger(),
    TensorboardVisualizerCallback(tb_dir),
    ModelSaverCallback(MODELS_DIR, epochs=args.epoch, every_n_epoch=args.save_every),
    MyReduceLROnPlateau(optimizer, loss_step="valid", factor=0.5, verbose=True, patience=args.patience)
]

learner = Learner(ClassifierCore(model, optimizer, loss), use_cuda=args.cuda)
learner.train(args.epoch, metrics, train_loader, test_loader, callbacks=callbacks)
