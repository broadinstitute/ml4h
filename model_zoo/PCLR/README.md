# PCLR (Patient Contrastive Learning of Representations)

PCLR is a pre-training strategy that yields a neural network that extracts representations of ECGs.
The representations are designed to be used in linear models without finetuning the network.
The pre-print is available [here](https://arxiv.org/abs/2104.04569).
This readme shows how to load a model trained on over three million ECGs using PCLR.

## Requirements
This code was tested using python 3.7.
It can be used using virtual env.
```bash
python3.7 -m venv env
source env/bin/activate
pip install -r requirements.txt
python -i get_representations.py  # test the setup worked
>>> test_get_representations()
```

## Usage
You can get ECG representations using [get_representations.py](./get_representations.py).
`get_representations.get_representations` builds `N x 320` ECG representations from `N` ECGs.

The model expects 10s 12-lead ECGs with a specific lead order and interpolated to be 4,096 samples long.
[preprocess_ecg.py](./preprocess_ecg.py) shows how to do the pre-processing.

## Lead I PCLR
We also provide a PCLR model using only lead I of the ECG at [PCLR_lead_I.h5](./PCLR_lead_I.h5).
It was trained with the same settings as the full 12-lead model except
the model only takes lead I of the ECG as input.

## Alternative save format
The newer keras saved model format is available for the 12-lead and lead I models at [PCLR](./PCLR)
and [PCLR_lead_I](./PCLR_lead_I).
