# ECG2HF-Subtype — Deep Learning to Predict Incident Heart Failure Subtypes

This directory contains the model and code for predicting the 10-year risk of incident heart failure with reduced ejection fraction (HFrEF) and heart failure with preserved ejection fraction (HFpEF) from 12-lead resting electrocardiograms (ECGs), as described in our [Journal of Cardiac Failure paper](https://doi.org/10.1016/j.cardfail.2026.08.022).

ECG2HF-Subtype is a deep learning model developed using a previously validated ECG-based heart failure prediction architecture. The model provides subtype-specific risk predictions to support identification of individuals at elevated risk of developing HFrEF or HFpEF.

## Model Inputs and Outputs

**Input:**
- **Modality:** 12-lead resting ECG
- **Expected shape:** `(batch, time, leads)` — confirm against released model
- **Sampling rate:** Confirm against released model
- **Normalization:** Confirm against training preprocessing

**Outputs:**
- **10-year risk prediction for incident HFrEF** (LVEF <50%)
- **10-year risk prediction for incident HFpEF** (LVEF ≥50%)

*Note: The exact output structure, including any auxiliary prediction heads, should be verified against the released model.*

## Model Weights and Inference

The pretrained model weights will be available in this directory.

If stored using Git LFS, the model can be downloaded using:

```bash
git lfs pull --include model_zoo/ECG2HF-Subtype/<model_filename>
```

To load the model and generate predictions in a Jupyter notebook using the ML4H environment:

```python
import numpy as np
from tensorflow.keras.models import load_model

model = load_model("./<model_filename>", compile=False)

# Replace with a correctly preprocessed ECG
ecg = np.random.random((1, 5000, 12)).astype(np.float32)

prediction = model(ecg)

for name, pred in zip(model.output_names, prediction):
    print(f"{name}: {pred}")
```

The example illustrates the model-loading workflow. Input dimensions, preprocessing, output interpretation, and the final model filename must be confirmed before using the model for inference. Random inputs do not produce clinically meaningful predictions.

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

Values in parentheses represent 95% confidence intervals.

The model consistently outperformed age and sex alone for both heart failure subtypes.

In comparisons with a clinical model incorporating PREVENT-HF variables, ECG2HF-Subtype demonstrated higher discrimination for HFrEF, while the clinical model demonstrated modestly higher discrimination for HFpEF.

These findings suggest that deep learning applied to routine ECGs may enable subtype-specific heart failure risk stratification and support tailored prevention strategies.

## Model Architecture

ECG2HF-Subtype builds upon a previously validated convolutional neural network architecture for ECG-based heart failure prediction.

The model processes resting ECG waveforms to learn representations associated with subsequent heart failure risk, with separate predictions for incident HFrEF and HFpEF.

*Add architecture diagram and the exact output-head descriptions after confirming the final model implementation.*

## Citation

If you use ECG2HF-Subtype in your research, please cite:

Khurshid S, D'Souza V, Friedman SF, et al. Electrocardiogram-based artificial intelligence to predict incident heart failure with preserved versus reduced ejection fraction. *Journal of Cardiac Failure*. 2026.

https://doi.org/10.1016/j.cardfail.2026.08.022

## Related Models

- [ECG2HF](https://github.com/broadinstitute/ml4h/tree/master/model_zoo/ECG2HF) — Deep learning for incident heart failure prediction.
- [ECG2Stroke](https://github.com/broadinstitute/ml4h/tree/master/model_zoo/ECG2Stroke) — Deep learning for incident ischemic stroke prediction.
- [ML4H Repository](https://github.com/broadinstitute/ml4h) — Machine Learning for Health at the Broad Institute.

## Intended Use

ECG2HF-Subtype is intended for research applications involving ECG-based heart failure risk prediction. It has not been established as a standalone clinical diagnostic tool or validated for independent clinical decision-making.
