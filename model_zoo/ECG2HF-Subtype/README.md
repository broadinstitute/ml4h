# ECG2HF-Subtype — Deep Learning to Predict Incident Heart Failure Subtypes

This directory contains the model and code for predicting the 10-year risk of incident heart failure with reduced ejection fraction (HFrEF) and heart failure with preserved ejection fraction (HFpEF) from 12-lead resting electrocardiograms (ECGs), as described in our [Journal of Cardiac Failure paper](https://doi.org/10.1016/j.cardfail.2026.08.022).

ECG2HF-Subtype is a deep learning model developed using a previously validated ECG-based heart failure prediction architecture. The model provides subtype-specific risk predictions to support identification of individuals at elevated risk of developing HFrEF or HFpEF.

## Model Inputs and Outputs

**Input:**
- **Modality:** 12-lead resting ECG
- **Expected shape:** `(batch, time, leads)` = `(B, 5000, 12)`
- **Sampling rate:** 500 Hz (10 seconds => 5000 samples)

**Outputs:**
- **10-year risk prediction for incident HFrEF** (LVEF <50%)
- **10-year risk prediction for incident HFpEF** (LVEF ≥50%)
- **Survival curve prediction for incident heart failure** 
- **Survival curve prediction for death**
- **Sex classification**
- **Age regression**


## Model Weights and Inference


If stored using Git LFS, the model can be downloaded using:

```bash
git lfs pull --include model_zoo/ECG2HF-Subtype/ecg_hfsubtype_wce_osp_8_10_v2024_03_20_20_21_16.keras
```

To load the model and generate predictions in a Jupyter notebook using the ML4H environment:

```python
import numpy as np
from tensorflow.keras.models import load_model

model = load_model("./ecg_hfsubtype_wce_osp_8_10_v2024_03_20_20_21_16.h5", compile=False)

ecg = np.random.random((1, 5000, 12)).astype(np.float32)

prediction = model(ecg)
```  

The model has 6 output heads: 10-year risk prediction for incident HF subtype defined using NLP definition, 10-year risk prediction for incident HF subtype defined using ICD codes, the survival curve prediction for incident heart failure, the survival curve prediction for death, sex classification and age regression. Those outputs can be accessed with:  

```python
for name, pred in zip(model.output_names, prediction):
    otm = output_tensormaps[name]
    if otm.is_categorical():
        if (otm.name == 'hf_nlp_lvef_type'):
            print(f'NLP defined HFrEF Risk prediction is: {pred[0, 1]}')
            print(f'NLP defined HFpEF Risk prediction is: {pred[0, 2]}')
        elif (otm.name == 'primary_lvef_type'):
            print(f'ICD code defined HFrEF Risk prediction is: {pred[0, 1]}')
            print(f'ICD code defined HFpEF Risk prediction is: {pred[0, 2]}')
        else:
            print(f'{otm} prediction is {pred[0,1]}')
    elif otm.is_survival_curve():
        intervals = otm.shape[-1] // 2
        days_per_bin = 1 + (2*otm.days_window) // intervals
        predicted_survivals = np.cumprod(pred[:, :intervals], axis=1)
        print(f'Stroke Risk {otm} prediction is: {str(1 - predicted_survivals[0, -1])}')
    else:
        print(f'{otm} prediction is {pred}')
```



## Study Design

ECG2HF-Subtype was developed using ECGs from 98,190 individuals at Massachusetts General Hospital (MGH) and evaluated in three independent test sets comprising individuals without prior heart failure:

- **MGH:** 16,739 individuals
- **Brigham and Women's Hospital (BWH):** 75,195 individuals
- **Beth Israel Deaconess Medical Center (BIDMC):** 30,133 individuals

Incident heart failure events were identified using a previously validated natural language processing model. Heart failure subtypes were determined based on left ventricular ejection fraction (LVEF) measured proximate to the incident heart failure diagnosis.

## Model Performance

ECG2HF-Subtype demonstrated consistent discrimination of incident HFrEF and HFpEF across all three test sets.

**10-year time-dependent AUROC:**

| Test Set | HFrEF | HFpEF |
|---|---|---|
| MGH | 0.89 (0.86–0.91) | 0.83 (0.81–0.85) |
| BWH | 0.88 (0.87–0.89) | 0.79 (0.77–0.80) |
| BIDMC | 0.88 (0.86–0.89) | 0.81 (0.80–0.82) |


The model consistently outperformed age and sex alone for both heart failure subtypes.

In comparisons with a clinical model incorporating PREVENT-HF variables, ECG2HF-Subtype demonstrated higher discrimination for HFrEF, while the clinical model demonstrated modestly higher discrimination for HFpEF.

These findings suggest that deep learning applied to routine ECGs may enable subtype-specific heart failure risk stratification and support tailored prevention strategies.

## Model Architecture

ECG2HF-Subtype builds upon a previously validated convolutional neural network architecture for ECG-based heart failure prediction.

The model processes ECG waveforms to learn representations associated with subsequent heart failure risk, with separate predictions for incident HFrEF and HFpEF.

*Add architecture diagram*





## Related Model:

- [ECG2HF](https://github.com/broadinstitute/ml4h/tree/master/model_zoo/ECG2HF) — Deep learning for incident heart failure prediction.


