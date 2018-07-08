# PyTorch CDSSM implementation for One-Shot Named Entity Linking

## Run on [paperspace](https://paperspace.com)

Download data to storage

```
paperspace jobs create --container Test-Container --machineType C2 --command "cd data; bash -x download_kaggle.sh"
```

Run learning

```
paperspace jobs create --container "ufoym/deepo:pytorch-py36" --machineType P4000 --command "bash -x run.sh"
```

