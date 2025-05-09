## ECG To Heart Failure (ECG2HF)
This directory contains the model file for predicting incident heart failure from ECGs.  
The model is a multitask convolutional neural network the primary task is to predict the survival curve for incident heart failure as defined by a previously [published NLP model](https://pubmed.ncbi.nlm.nih.gov/36939660/).
The four auxiliary task are prediction of incident heart failure as defined by primary ICD code, incident mortality, age regression, and sex classification.
The model expects a 10 second 12-lead resting ECG as input sampled at 500Hz with voltages normalized per-ECG to have a mean of 0 and standard deviation of 1.


The raw model files are stored using `git lfs` so you must have `git` and `git lfs` installed.
Then the full ~33MB network can be localized with:
```bash
git lfs pull --include model_zoo/ECG2HF/ecg_5000_hf_quintuplet_dropout_v2023_04_17.keras
```

The model can be loaded and the predictions accessed using the following code:
```python
import numpy as np
from tensorflow.keras.models import load_model
from ml4h.tensormap.ukb.demographics import age_in_days, is_male_mgb
from ml4h.tensormap.ukb.survival import mgb_hf_nlp_wrt_instance2, mgb_hf_primary_wrt_instance2, mgb_death_wrt_instance2

output_tensormaps = {tm.output_name(): tm for tm in [is_male_mgb, mgb_hf_primary_wrt_instance2, 
                                                     mgb_hf_nlp_wrt_instance2, mgb_death_wrt_instance2, age_in_days]}
model = load_model('./ecg_5000_hf_quintuplet_dropout_v2023_04_17.keras')
ecg = np.random.random((1, 5000, 12))
prediction = model(ecg)

for name, pred in zip(model.output_names, prediction):
    otm = output_tensormaps[name]
    if otm.is_survival_curve():
        intervals = otm.shape[-1] // 2
        days_per_bin = 1 + otm.days_window // intervals
        predicted_survivals = np.cumprod(pred[:, :intervals], axis=1)
        print(f'AF Risk {otm} prediction is: {str(1 - predicted_survivals[0, -1])}')
    else:
        print(f'{otm} prediction is {pred}')
```