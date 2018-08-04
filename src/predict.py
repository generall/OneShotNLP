import nltk
import argparse

from torchlite.torch.train_callbacks import ModelSaverCallback

from model.embedding import ModelVectorizer, OnDiskVectorizer
from model.arc2 import ARC2, PreConv


def tokenizer(text, alpha_only=True):  # create a tokenizer function
    words = [tok for tok in nltk.word_tokenize(text) if (not alpha_only or tok.isalpha())]
    return words


parser = argparse.ArgumentParser(description='ARC2 inference')

parser.add_argument('--model', dest='model', help='path to read model')
parser.add_argument('--input-file', dest='input_file')


parser.add_argument('--cuda', type=bool, default=False)
parser.add_argument('--emb-path', type=str, default=None)

parser.add_argument('--mtx-path', type=str, default=None)
parser.add_argument('--meta-path', type=str, default=None)



net_params = {
    'preconv': True,
    'word_emb_sizes': [300],
    'preconv_size': [300],
    'matrix_depth': [120],
    'conv_depth': [120, 60, 60],
    'out_size': [60]
}

args = parser.parse_args()

if args.emb_path:
    vectorizer = ModelVectorizer(model_path=args.emb_path)
else:
    vectorizer = OnDiskVectorizer(mtx_path=args.mtx_path, meta_path=args.meta_path)


preconv_size = net_params['preconv_size'] if net_params['preconv'] else None


preconv = PreConv(
    word_emb_sizes=net_params['word_emb_sizes'],
    sent_conv_size=preconv_size,
    dropout=0.0,
    window=2
)

model = ARC2(
    vectorizer=None,
    preconv=preconv,
    matrix_depth=net_params['matrix_depth'],
    conv_depth=net_params['conv_depth'],
    out_size=net_params['out_size'],
    window=2,
    dropout=0.0
)

ModelSaverCallback.restore_model_from_file(model, args.model, load_with_cpu=(not args.cuda))

model.eval()

with open(args.input_file) as fd:
    for line in fd:
        line = line.strip()
        sent_a, sent_b = line.split('|')
        sent_a_token = tokenizer(sent_a) + [' '] * 6
        sent_b_token = tokenizer(sent_b) + [' '] * 6

        sent_a_vect = vectorizer.convert([sent_a_token])
        sent_b_vect = vectorizer.convert([sent_b_token])

        score = model.forward(sent_a_vect, sent_b_vect)

        print(line, score[0][1].item())




