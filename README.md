# ml4cvd
Machine Learning for CardioVascular Disease - MGH/MIT edition!

## Contents
- Setup
- Run scripts
- [Tensorize ECGs](documentation/tensorize_ecgs.md)
- Contribute

## Setup
TODO

### Conda

### Docker

## Modes
TODO

## Run scripts
Run scripts are stored in [this Dropbox folder](https://www.dropbox.com/sh/hjz7adj01x1erfs/AABnZifp1mUqs7Z_26zm4ly9a?dl=0).

### Script dispatcher

To distribute training scripts across bootstraps and GPUs, use [`scripts/dispatch.py`](https://github.com/aguirre-lab/ml/blob/er_dispatcher/scripts/dispatch.py):

```zsh
python scripts/dispatch.py \
--gpus 0-3 \
--bootstraps 0-9 \
--scripts \
    ~/dropbox/ml4cvd_run_scripts/sts-ecg/train-simple.sh
    ~/dropbox/ml4cvd_run_scripts/sts-ecg/train-varied.sh
    ~/dropbox/ml4cvd_run_scripts/sts-ecg/train-deeper.sh
```

## Contribute

Submit a new feature request, bug report, etc. by creating a [new issue with the ml4cvd template](https://github.com/aguirre-lab/ml/issues/new/choose).

TODO discuss PR and review norms
