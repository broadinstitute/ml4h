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

### Use git LFS to localize the model file
Make sure git lfs is installed on your system and then from within the ml4h repo on your machine run:
`git lfs pull --include model_zoo/PCLR/PCLR.h5` 

### Example inference with UKB ECGs tensorized with ML4H
This code snippet produces a CSV of a latent space from ECGs stored in HD5s.
The HD5s can be made from XMLs by with [ml4h/tensorize/tensor_writer_ukbb.py](ml4h/tensorize/tensor_writer_ukbb.py).
```python
import os
from typing import List, Dict

import csv
import h5py
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model, Model

from preprocess_ecg import process_ecg, LEADS

tensors = '/mnt/disks/ecg-rest-68k-tensors/2023_09_17/' # Replace this with path to your tensors
model = load_model("./PCLR.h5")
model.summary()

leads = [
    'I', 'II', 'III', 'aVR', 'aVL', 'aVF',
    'V1', 'V2', 'V3', 'V4', 'V5', 'V6',
]

latent_dimensions = 320

with open('./pclr_ukb_inferences_v2023_10_24.tsv', mode='w') as inference_file:
    inference_writer = csv.writer(inference_file, delimiter='\t', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    header = ['sample_id']
    header += [f'latent_{i}' for i in range(latent_dimensions)]
    inference_writer.writerow(header)
    
    for i,f in enumerate(sorted(os.listdir(tensors))):
        with h5py.File(f'{tensors}{f}', 'r') as hd5:
            ecg = np.zeros((1, 4096, 12))
            for k,l in enumerate(leads):
                lead = np.array(hd5[f'/ukb_ecg_rest/strip_{l}/instance_0'])
                interpolated_lead = np.interp(np.linspace(0, 1, 4096),
                                              np.linspace(0, 1, lead.shape[0]),
                                              lead,
                                             )
                ecg[0, :, k] = interpolated_lead/1000
            ls = model(ecg)
            sample_id = os.path.basename(f).replace('.hd5', '')
            csv_row = [sample_id]
            csv_row += [f'{ls[0, i]}' for i in range(latent_dimensions)]
            inference_writer.writerow(csv_row)
        if i % 500 == 0:
            print(f'ECGs found in {i} files, last tensor was {f}')
```


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
