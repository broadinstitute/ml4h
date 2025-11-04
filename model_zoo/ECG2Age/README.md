# ECG2Age — Age Prediction from 12‑Lead ECG (Keras)

A lightweight Keras/TensorFlow model that predicts chronological age from a standard 12‑lead ECG.

## Table of Contents

* [Model card](#model-card)

  * [Inputs](#inputs)
  * [Outputs](#outputs)
  * [Performance](#performance)
* [Getting started](#getting-started)

  * [Environment](#environment)
  * [Repository layout](#repository-layout)
* [Model files](#model-files)
* [Load & run inference](#load--run-inference)

---

## Model card

### Inputs

* **Modality**: 12‑lead resting ECG
* **Expected shape**: `(batch, time, leads)` = `(B, 5000, 12)` by default

  * Sampling rate: **500 Hz** (10 seconds ⇒ 5000 samples)
  * If your data is `(B, 12, 5000)`, set `--channels_first` or transpose before feeding the model.
* **Dtype / range**: `float32` normalized per‑lead (e.g., z‑score or min‑max).
  Provide your normalization in `data/processing.py`.

> Update the shape/sampling rate above if your dataset differs.

### Outputs

* **Target**: chronological age in **years** (scalar per record)
* **Prediction head**: single linear unit with optional activation clipping (e.g., `ReLU` at 0 years)
* **Loss**: MAE or Huber (configurable)

### Performance

Please benchmark on a **held‑out test set** (subject‑wise split). Example table with placeholders:

| Metric      |    Test          |
| ----------- | ---------------: |
| R²          |     0.3056011687 |
| Pearson r   |     0.5528120555 |

Add a calibration plot and error vs age plot if possible (see `notebooks/metrics.ipynb`).

---

## Getting started

### Environment

```bash
# Python ≥3.10 recommended
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
# or
pip install tensorflow==2.16.* keras==3.* numpy scipy scikit-learn matplotlib h5py
```

### Repository layout

```
.
├── model_zoo/ECG2Age
│   ├── ECG2Age.keras        # Keras SavedModel (via Keras 3)
    ├── encoder_median.keras # Encoder part of ECG2Age model 
    ├── decoder_instance_age.keras  # Decoder part of ECG2Age model
    ├── merger.keras          #Encoder decoder merger intermediate model

```

---

## Load & run inference

### Python API

```python
import numpy as np
import tensorflow as tf
from keras import ops

# Load model (Keras 3 format)
model = tf.keras.models.load_model("model_zoo/ECG2Age/ECG2Age.keras", compile=False)

# Example input: batch of 8 ECGs, 5000 samples, 12 leads
x = np.random.randn(8, 5000, 12).astype("float32")
# Optional: apply same normalization used during training
# x = normalize_batch(x)

pred_age = model.predict(x, verbose=0)  # shape (8, 1)
print(pred_age.squeeze())
```
