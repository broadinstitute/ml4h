## ECG2AF: Deep Learning to Predict Incident Atrial Fibrillation
This directory contains models and code for predicting incident atrial fibrillation from 12 lead resting ECGs, as described in our 
[Circulation paper](https://www.ahajournals.org/doi/full/10.1161/CIRCULATIONAHA.121.057480).

To load the 12 lead model in a jupyter notebook (running with the ml4h docker or python library installed) see the [example](./ecg2af_infer.ipynb) or run: 
```python
output_tensormaps = {tm.output_name(): tm for tm in [mgb_afib_wrt_instance2, age_2_wide, af_dummy, sex_dummy3]}
custom_dict = get_custom_objects([mgb_afib_wrt_instance2, age_2_wide, af_dummy, sex_dummy3])
model = load_model('./ecg_5000_survival_curve_af_quadruple_task_mgh_v2021_05_21.h5', custom_objects=custom_dict)
ecg = np.random.random((1, 5000, 12))
prediction = model(ecg)
```
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
    --output_tensors survival.mgb_afib_wrt_instance2 demographics.age_2_wide \
                     demographics.af_dummy demographics.sex_dummy \
     --tensormap_prefix ml4h.tensormap.ukb \
     --id ecg2afib_quadruple_task_inference \
     --output_folder /path/to/ml4h_runs/ \
     --model_file /path/to/ml4h/model_zoo/ECG2AF/ecg_5000_survival_curve_af_quadruple_task_mgh_v2021_05_21.h5
```

The model weights for the main model which performs incident atrial fibrillation prediction as well as the auxiliary tasks of
age regression, sex classification and prevalent (at the time of ECG) atrial fibrillation:
[ecg_5000_survival_curve_af_quadruple_task_mgh_v2021_05_21.h5](./ecg_5000_survival_curve_af_quadruple_task_mgh_v2021_05_21.h5)

We also include single lead models for lead strip I:[strip_I_survival_curve_af_v2021_06_15.h5](./strip_I_survival_curve_af_v2021_06_15.h5)
and II: [strip_II_survival_curve_af_v2021_06_15.h5](./strip_II_survival_curve_af_v2021_06_15.h5)

### Study Design
Flow chart of study design
![Flow chart of study design](./study_design.jpg)
### Performance
Risk stratification model comparison
![Risk stratification model comparison](./km.jpg)
### Salience
Salience and Median waveforms from predicted risk extremes.
![Salience and Median waveforms](./salience.jpg)
### Architecture
1D Convolutional neural net architecture
![Convolutional neural net architecture](./architecture.png)
