# PyTorch implementation of One-Shot Named Entity Linking

Designed for [OneShot-wikilinks](https://www.kaggle.com/generall/oneshotwikilinks) dataset.

Training with [fastText](https://fasttext.cc/docs/en/pretrained-vectors.html) embeddings

```bash
cd src
python train_arcii.py --cuda=1 --epoch 10 --dropout 0.5\
                      --netsize 120 --parallel 10\
                      --run fasttext_arc2\
                      --train-data ../data/full_data_train.tsv\
                      --valid-data ../data/full_data_valid.tsv\
                      --save-every 1 --read-size 1000 --batch-size 50\
                      --lr 0.0005 --patience 4 --emb-size 300\
                      --cycles 1 --preconv 1 --emb-path '../data/wiki.en.bin' |& tee run.sh.log
```

Validation accuracy: `0.88`

Eval

```bash
python predict.py --emb-path ../data/wiki.en.bin --model ../data/models/ARC2_best.pth --input <input_file>
```


## Run on [paperspace](https://paperspace.com)

Download data to storage

```
paperspace jobs create --container Test-Container --machineType C2 --command "cd data; bash -x download_kaggle.sh"
```

Run learning

```
paperspace jobs create --container "ufoym/deepo:pytorch-py36" --machineType P4000 --command "bash -x run.sh"
```


