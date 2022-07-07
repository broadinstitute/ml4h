# PCLR (Patient Contrastive Learning of Representations)

PCLR is a pre-training strategy that yields a neural network that extracts representations of ECGs.
The representations are designed to be used in linear models without finetuning the network.
The paper is available in PLOS comp bio [here](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1009862).
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
### Getting ECG representations
You can get ECG representations using [get_representations.py](./get_representations.py).
`get_representations.get_representations` builds `N x 320` ECG representations from `N` ECGs.

The model expects 10s 12-lead ECGs with a specific lead order and interpolated to be 4,096 samples long.
[preprocess_ecg.py](./preprocess_ecg.py) shows how to do the pre-processing.

### Building un-trained PCLR and comparison models

You can get compiled, but un-trained models with the hyperparameters selected in our training set.
`python -i build_model.py`
```python
pclr_model = PCLR_model()
clocs_model = CLOCS_model()
CAE = CAE_model()
ribeiro_r = ribeiro_r_model()
```
`build_model.py` uses code from [the google research implementation of SimCLR](https://github.com/google-research/simclr/)
and [the official implementation](https://github.com/antonior92/automatic-ecg-diagnosis) of "Automatic diagnosis of the 12-lead ECG using a deep neural network",
Ribeiro et al 2020.

## Lead I PCLR
We also provide a PCLR model using only lead I of the ECG at [PCLR_lead_I.h5](./PCLR_lead_I.h5).
It was trained with the same settings as the full 12-lead model except
the model only takes lead I of the ECG as input.

## Lead II PCLR
[Lead II PCLR](./PCLR_lead_II.h5) is like lead I PCLR except it was trained with all ECGs sampled to 250Hz.

## Alternative save format
The newer keras saved model format is available for the 12-lead and single lead models at [PCLR](./PCLR)
and [PCLR_lead_I](./PCLR_lead_I) and [PCLR_lead_II](./PCLR_lead_II).
