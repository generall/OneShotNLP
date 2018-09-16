
import nltk
import argparse

from torchlite.torch.train_callbacks import ModelSaverCallback

from model.embedding import ModelVectorizer, OnDiskVectorizer
from model.siames import Matcher


def tokenizer(text, alpha_only=True):  # create a tokenizer function
    tokens = [tok for tok in nltk.word_tokenize(text) if (not alpha_only or tok.isalpha())]
    if len(tokens) <= 1:
        tokens += ['xxx', 'xxx']
    return tokens


parser = argparse.ArgumentParser(description='CDSSM tele2 matcher')

parser.add_argument('--model', dest='model', help='path to read model')
parser.add_argument('--input-file', dest='input_file')


parser.add_argument('--cuda', type=bool, default=False)
parser.add_argument('--emb-path', type=str, default=None)

parser.add_argument('--mtx-path', type=str, default=None)
parser.add_argument('--meta-path', type=str, default=None)


net_params = {
    'preconv': True,
    'word_emb_sizes': 300,
    'conv_sizes': [200],
    'out_size': [200, 100]
}


args = parser.parse_args()

if args.emb_path:
    vectorizer = ModelVectorizer(model_path=args.emb_path)
else:
    vectorizer = OnDiskVectorizer(mtx_path=args.mtx_path, meta_path=args.meta_path)

model = Matcher(
    word_emb_sizes=net_params['word_emb_sizes'],
    conv_sizes=net_params['conv_sizes'],
    out_size=net_params['out_size'],
    dropout=0.0,
)

ModelSaverCallback.restore_model_from_file(model, args.model, load_with_cpu=(not args.cuda))

model.eval()

with open(args.input_file) as fd:
    for line in fd:
        line = line.strip()
        sent_b = line
        sent_b_token = tokenizer(sent_b)
        if len(sent_b_token) <= 2:
            sent_b_token += ['xxx']
        sent_b_vect = vectorizer.convert([sent_b_token])
        vect = model.cdssm_b.process_sentences(sent_b_vect)

        print(" ".join(map(str, vect[0].detach().numpy().tolist())))




