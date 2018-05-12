#!/usr/bin/env bash

# Download dependencies and run


pip install -r requirements.txt

python -c "import nltk; nltk.download('punkt')"
#python -m spacy download en

cd src/

python train.py
