#!/usr/bin/env bash

# paperspace jobs create --container Test-Container --machineType C2 --command "cd data; bash -x download_kaggle.sh"

pip install kaggle

mkdir -p ~/.kaggle

cp kaggle.json ~/.kaggle/kaggle.json

chmod 600 ~/.kaggle/kaggle.json

kaggle datasets download -d generall/oneshotwikilinks -w

mv oneshotwikilinks.zip /storage
