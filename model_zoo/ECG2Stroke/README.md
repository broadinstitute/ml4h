# ECG2Stroke — Deep Learning to Predict Incident Ischemic Stroke

This directory contains models and code for predicting incident ischemic stroke from 12-lead resting ECGs, as described in <mark>INSERT LINK HERE</mark>.

**Input:**
* **Modality**: 12‑lead resting ECG
  * **Expected shape**: `(batch, time, leads)` = `(B, 5000, 12)`

    * Sampling rate: **500 Hz** (10 seconds ⇒ 5000 samples)
  * **Normalization**: Z-score normalized per‑lead (i.e., mean of 0 and standard deviation of 1).

**Outputs:**
* **Survival curve prediction for incident ischemic stroke**
* **Survival curve prediction for death**
* **Sex classification**
* **Age regression**
* **Classification of atrial fibrillation at the time of ECG**

The raw model files are stored using `git lfs` so you must have it installed and localize the full ~225MB file with:
```bash
git lfs pull --include model_zoo/ECG2Stroke/ecg2stroke_dropout_2024_10_04_10_49_43.h5
```

To load the model in a jupyter notebook (running with the ml4h docker), run:

```python
import numpy as np
from tensorflow.keras.models import load_model
from ml4h.tensormap.ukb.demographics import sex_dummy1, age_in_days, af_dummy2
from ml4h.tensormap.ukb.survival import mgb_stroke_wrt_instance2, mgb_death_wrt_instance2

output_tensormaps = {tm.output_name(): tm for tm in [mgb_stroke_wrt_instance2, mgb_death_wrt_instance2, 
                                                     sex_dummy1, age_in_days, af_dummy2]}
model = load_model('./ecg2stroke_dropout_2024_10_04_10_49_43.h5')
ecg = np.random.random((1, 5000, 12))
prediction = model(ecg)
```
If above does not work you may need to use an absolute path in `load_model`.

The model has 5 output heads: the survival curve prediction for incident ischemic stroke, the survival curve prediction for death, sex classification, age regression, and classification of atrial fibrillation at the time of ECG.  Those outputs can be accessed with:
```python
for name, pred in zip(model.output_names, prediction):
    otm = output_tensormaps[name]
    if otm.is_survival_curve():
        intervals = otm.shape[-1] // 2
        days_per_bin = 1 + (2*otm.days_window) // intervals
        predicted_survivals = np.cumprod(pred[:, :intervals], axis=1)
        print(f'Stroke Risk {otm} prediction is: {str(1 - predicted_survivals[0, -1])}')
    else:
        print(f'{otm} prediction is {pred}')
```


To perform command line inference with this model run:
```bash
  python /path/to/ml4h/ml4h/recipes.py \
    --mode infer \
    --tensors /path/to/tensors/ \
    --input_tensors ecg.ecg_rest_mgb \
    --output_tensors survival.mgb_stroke_wrt_instance2 survival.mgb_death_wrt_instance2 \
                     demographics.sex_dummy demographics.age_in_days demographics.af_dummy \
     --tensormap_prefix ml4h.tensormap.ukb \
     --id ecg2stroke_dropout_task_inference \
     --output_folder /path/to/ml4h_runs/ \
     --model_file /path/to/ml4h/model_zoo/ECG2Stroke/ecg2stroke_dropout_2024_10_04_10_49_43.h5'
```

### Study flow diagram
<div style="padding: 10px; background-color: white; display: inline-block;">
    <img src="./ecg2stroke_study_design.png" alt="Study flow diagram" />
</div>

### Performance
A) Smoothed calibration curves depicting predicted versus observed event probabilities for 10-year incident stroke for ECG2Stroke in MGH, BWH, and BIDMC test sets. Diagonal dashed line indicates perfect calibration. Curves are obtained using restricted cubic spliness22. B-D) Stroke-free survival stratified by quintiles of ECG2Stroke predicted risk for 10-year incident stroke in MGH, BWH, and BIDMC test sets. Transparent bands indicate 95% confidence intervals for survival probability. 
<div style="padding: 10px; background-color: white; display: inline-block;">
    <img src="./ecg2stroke_performance.png" alt="Calibration and risk stratification for 10-year incident stroke prediction" />
</div>

### Salience
<div style="padding: 10px; background-color: white; display: inline-block;">
Saliency maps of ECG2Stroke demarcating regions of the ECG waveform having the greatest influence on stroke risk predictions. Blue shades depict the magnitude of the gradient of predicted stroke risk with respect to the ECG waveform amplitude, where darker shades illustrate regions of the waveform exerting greater salience or influence on stroke risk predictions. Saliency was averaged over 7080 random samples from the Brigham and Women’s Hospital (BWH) test set . The black waveform depicts the median waveform in each lead among the 7080 individuals. <div style="padding: 10px; background-color: white; display: inline-block;">
    <img src="./ecg2stroke_salience.png" alt="ECG Salience" />
</div>

### Architecture
A schematic of the neural network architecture. The model takes 10 seconds of 12-lead ECG waveform data as input, which is processed through a series of convolutional layers. The resulting learned features are passed to fully-connected layers to produce an estimate of time to stroke (primary) as well as predictions of time to death, age, sex, and presence of AF in the ECG diagnostic statement (secondary). Arrows indicate the flow of data between layers. Conv1D, one-dimensional convolution, MaxPooling1D, one-dimensional maximum pooling.
<div style="padding: 10px; background-color: white; display: inline-block;">
    <img src="./ecg2stroke_architecture.png" alt="Neural network architecture" />
</div>
