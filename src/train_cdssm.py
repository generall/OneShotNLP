"""
This script should perform training of the CDSSM model
"""
import argparse
import datetime

import os

import torch

import nltk
import torch.optim as optim
from torchlite.torch.learner import Learner
from torchlite.torch.learner.cores import ClassifierCore
from torchlite.torch.train_callbacks import TensorboardVisualizerCallback, ModelSaverCallback, ReduceLROnPlateau

from config import TB_DIR, MODELS_DIR
from model.embedding import EmbeddingVectorizer, FastTextEmbeddingBag, ModelVectorizer
from model.loss import AccuracyMetric, RocAucMetric
from model.siames import Siames
from utils.loader import MentionsLoader, EmbeddingMentionLoader, WikiLinksReader
from utils.loggers import ModelParamsLogger


def tokenizer(text, alpha_only=True):  # create a tokenizer function
    return [tok for tok in nltk.word_tokenize(text) if (not alpha_only or tok.isalpha())]


parser = argparse.ArgumentParser(description='Train CDSSM model')

parser.add_argument('--train-data', dest='train_data', help='path to train data', default="")
parser.add_argument('--valid-data', dest='valid_data', help='path to valid data', default="")

parser.add_argument('--restore-model', dest='restore_model',
                    help='path to saved model')  # default=os.path.join(MODELS_DIR, 'Siames_epoch-150.pth'))

parser.add_argument('--epoch', type=int, default=100)
parser.add_argument('--save-every', type=int, default=10)
parser.add_argument('--read-size', type=int, default=250)
parser.add_argument('--batch-size', type=int, default=1000)
parser.add_argument('--cuda', type=bool, default=False)
parser.add_argument('--parallel', type=int, default=0)
parser.add_argument('--patience', type=int, default=10)
parser.add_argument('--run', default='none', help='name of current run for tensorboard')
parser.add_argument('--lr', type=float, default=1e-2)
parser.add_argument('--weight_decay', type=float, default=0)
parser.add_argument('--netsize', type=int, default=10)
parser.add_argument('--emb-size', type=int, default=300)
parser.add_argument('--dropout', type=float, default=0.0)
parser.add_argument('--cycles', type=int, default=1)
parser.add_argument('--emb-path', type=str, default=None)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--torch-seed', type=int, default=42)


args = parser.parse_args()

vectorizer = EmbeddingVectorizer(ModelVectorizer(model_path=args.emb_path))

batch_reader_train = WikiLinksReader(
    filename=args.train_data,
    read_size=args.read_size,
    batch_size=args.batch_size,
)

train_loader = EmbeddingMentionLoader(
    batch_reader=batch_reader_train,
    tokenizer=tokenizer,
    parallel=args.parallel,
    vectorizer=vectorizer,
)

batch_reader_test = WikiLinksReader(
    filename=args.valid_data,
    read_size=args.read_size,
    batch_size=args.batch_size,
)

test_loader = EmbeddingMentionLoader(
    batch_reader=batch_reader_test,
    tokenizer=tokenizer,
    parallel=args.parallel,
    vectorizer=vectorizer,
)

loss = torch.nn.BCELoss()


model = Siames(
    word_emb_sizes=args.emb_size,
    conv_sizes=[int(args.netsize)],
    out_size=[args.netsize, args.netsize // 2],
    dropout=args.dropout,
)


if args.restore_model:
    ModelSaverCallback.restore_model_from_file(model, args.restore_model, load_with_cpu=(not args.cuda))


optimizer = optim.Adam(filter(lambda x: x.requires_grad, model.parameters()), lr=args.lr,
                       weight_decay=args.weight_decay)

run_name = args.run + '-' + datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")

tb_dir = os.path.join(TB_DIR, run_name)
if not os.path.exists(tb_dir):
    os.mkdir(tb_dir)

metrics = [
    AccuracyMetric(),
    RocAucMetric()
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
