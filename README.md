# ml4cvd
Machine Learning for CardioVascular Disease

## Setup
TODO

### Conda

### Docker

## Modes
TODO

## Run scripts
Run scripts are stored in [this Dropbox folder](https://www.dropbox.com/sh/hjz7adj01x1erfs/AABnZifp1mUqs7Z_26zm4ly9a?dl=0).

### Script dispatcher

To train models across several bootstrap samples and GPUs, use [`dispatch.py`](https://github.com/aguirre-lab/ml/blob/er_dispatcher/scripts/dispatch.py).

For example, to run several training scripts across four GPUs and ten bootstraps:

```zsh
python dispatch.py \
--gpus 0-3 \
--bootstraps 0-9 \
--scripts \
    ~/dropbox/ml4cvd_run_scripts/sts-ecg/train-simple.sh
    ~/dropbox/ml4cvd_run_scripts/sts-ecg/train-varied.sh
    ~/dropbox/ml4cvd_run_scripts/sts-ecg/train-deeper.sh
```

## Contributing

Submit a new feature request, bug report, etc. by creating a [new issue with the ml4cvd template](https://github.com/aguirre-lab/ml/issues/new/choose).

TODO discuss PR and review norms
