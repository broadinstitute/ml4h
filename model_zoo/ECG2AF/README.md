## ECG2AF: Deep Learning to Predict Incident Atrial Fibrillation
This directory contains models and code for predicting incident atrial fibrillation from 12 lead resting ECGs, as described in our 
[Circulation paper](https://www.ahajournals.org/doi/full/10.1161/CIRCULATIONAHA.121.057480).

The raw model files are stored using `git lfs` so you must have it installed and localize the full ~200MB files with:
```bash
git lfs pull --include model_zoo/ECG2AF/*keras
```

To load the 12 lead model in a jupyter notebook (running with the ml4h docker or python library installed) see the [example](./ecg2af_infer.ipynb) or run:

```python
import numpy as np
from tensorflow.keras.models import load_model
from ml4h.tensormap.ukb.demographics import age_in_days, af_dummy2, sex_dummy1
from ml4h.tensormap.ukb.survival import mgb_afib_wrt_instance2, mgb_death_wrt_instance2

output_tensormaps = {tm.output_name(): tm for tm in [mgb_afib_wrt_instance2, mgb_death_wrt_instance2, 
                                                     age_in_days, af_dummy2, sex_dummy1]}
model = load_model('./ecg2af_quintuplet_v2024_01_13.keras')
ecg = np.random.random((1, 5000, 12))
prediction = model(ecg)
```
If above does not work you may need to use an absolute path in `load_model`.

The model has 4 output heads: the survival curve prediction for incident atrial fibrillation, the classification of atrial fibrillation at the time of ECG, sex, and age regression.  Those outputs can be accessed with:
```python
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


To perform command line inference with this model run:
```bash
  python /path/to/ml4h/ml4h/recipes.py \
    --mode infer \
    --tensors /path/to/tensors/ \
    --input_tensors ecg.ecg_rest_mgb \
    --output_tensors survival.mgb_afib_wrt_instance2 survival.mgb_death_wrt_instance2 \
                     demographics.age_in_days demographics.af_dummy demographics.sex_dummy \
     --tensormap_prefix ml4h.tensormap.ukb \
     --id ecg2afib_quadruple_task_inference \
     --output_folder /path/to/ml4h_runs/ \
     --model_file /path/to/ml4h/model_zoo/ECG2AF/ecg2af_quintuplet_v2024_01_13.keras'
```

The model weights for the main model which performs incident atrial fibrillation prediction auxiliary tasks of
as well as incident mortality prediction, age regression, sex classification and prevalent (at the time of ECG) atrial fibrillation.

This model was updated in May 2025, the original model without mortality prediction was trained in 2021, and is still available here:
[ecg_5000_survival_curve_af_quadruple_task_mgh_v2021_05_21.h5](./ecg_5000_survival_curve_af_quadruple_task_mgh_v2021_05_21.h5)


We also include single lead models for lead/strip I: [strip_I_survival_curve_af_v2021_06_15.h5](./strip_I_survival_curve_af_v2021_06_15.h5)
and II: [strip_II_survival_curve_af_v2021_06_15.h5](./strip_II_survival_curve_af_v2021_06_15.h5)

### Study design
<div style="padding: 10px; background-color: white; display: inline-block;">
    <img src="./study_design.jpg" alt="Flow chart of study design" />
</div>

### Performance
Risk stratification model comparison
<div style="padding: 10px; background-color: white; display: inline-block;">
    <img src="./km.jpg" alt="Risk stratification model comparison" />
</div>

### Salience
Salience and Median waveforms from predicted risk extremes.
![Salience and Median waveforms](./salience.jpg)
### Architecture
1D Convolutional neural net architecture
![Convolutional neural net architecture](./architecture.png)
