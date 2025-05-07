## ECG To Heart Failure (ECG2HF)
This directory contains the model file for predicting incident heart failure from ECGs.

The raw model files are stored using `git lfs` so you must have `git` and `git lfs` installed and localize the full ~135MB autoencoder as well as the component decoder and encoder:
```bash
git lfs pull --include model_zoo/ECG2HF/ecg_5000_hf_quintuplet_dropout_v2023_04_17.keras
```
