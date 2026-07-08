## ECG PheWAS
This directory contains python notebooks and instructions to create the models and results from 
[this NPJ Digital Medicine Paper](https://www.nature.com/articles/s41746-024-01418-9).

The raw model files are stored using `git lfs` so you must have `git` and `git lfs` installed and localize the full ~135MB autoencoder as well as the component decoder and encoder:
```bash
git lfs pull --include model_zoo/ECG_PheWAS/*.h5
```

Our model expects ECG median waveforms with 600 milliVolt voltages across 12 leads as input and produces 
a 256 dimensional latent space encoding, as well as a reconstructed ECG with the same shape as the input. 
The notebook [ecg_write_biosppy_medians.ipynb](./ecg_write_biosppy_medians.ipynb) provides an example of creating these median waveforms from 10 second 12 lead ECGs.

The electrocardiogram (ECG) is an inexpensive and widely available diagnostic tool, and therefore has great potential 
to facilitate disease detection in large-scale populations. 
Both cardiac and noncardiac diseases may alter the appearance of the ECG, though the extent to which diseases across 
the human phenotypic landscape can be detected on the ECG remains unclear. 
We developed an autoencoder model that encodes and reconstructs ECG waveform data within a 
multidimensional latent space.
The ECG latent space model demonstrated a greater number of associations than ECG models using standard ECG intervals 
alone, and generally resulted in improvements in discrimination of diseases compared to models comprising 
only age, sex, and race. 
We further demonstrate how a latent space model can be used to generate disease-specific ECG waveforms and facilitate 
disease profiling for individual patients.

To create a model from scratch run:
```bash
  python /path/to/ml4h/ml4h/recipes.py \
    --mode train \
    --tensors /path/to/hd5_tensors/ \
    --output_folder /path/to/output/ \
    --tensormap_prefix ml4h.tensormap.ukb \
    --input_tensors ecg.ecg_biosppy_median_60bpm --output_tensors ecg.ecg_biosppy_median_60bpm \
    --encoder_blocks conv_encode --decoder_blocks conv_decode --activation mish --conv_layers 23 23 \
    --dense_blocks 46 --block_size 5 --dense_layers 256 --dense_normalize layer_norm \
    --batch_size 2 --epochs 96 --training_steps 128 --validation_steps 36 --test_steps 32 --patience 64 \
    --id ecg_median_autoencoder
```

Given this model, infer a latent space with:
```bash
python /path/to/ml4h/ml4h/recipes.py \
    --mode infer_encoders \
    --tensors /path/to/hd5_tensors/ \
    --output_folder /path/to/output/ \
    --tensormap_prefix ml4h.tensormap.ukb \
    --input_tensors ecg.ecg_biosppy_median_60bpm --output_tensors ecg.ecg_biosppy_median_60bpm \
    --model_file /path/to/ml4h/model_zoo/ECG_PheWAS/mgh_biosppy_median_60bpm_autoencoder_256d_v2022_05_21.h5 \
    --id ecg_median_autoencoder 
```

With this latent space and phecode diagnoses for the same cohort, the jupyter notebook 
[latent_space_phewas](./latent_space_phewas.ipynb)
allows you to conduct the PheWAS analysis.

![UKB PheWAS Plot](./ukb_phewas.png)
