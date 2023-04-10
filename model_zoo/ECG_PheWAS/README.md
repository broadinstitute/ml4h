## ECG PheWAS
This directory contains python notebooks and instructions to create the models and results from 
[this paper](https://www.medrxiv.org/content/10.1101/2022.12.21.22283757v1).

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

Our model expects ECG median waveforms with 600 centiVolt voltages across 12 leads as input and produces 
a 256 dimensional latent space encoding, as well as a reconstructed ECG with the same shape as the input.

To create a model from scratch run:
```bash
  python /path/to/ml4h/ml4h/recipes.py \
    --mode train \
    --tensors /path/to/hd5_tensors/ \
    --output_folder /path/to/output/ \
    --tensormap_prefix ml4h.tensormap.ukb \
    --input_tensors ecg.ecg_rest_median_raw_10 --output_tensors ecg.ecg_rest_median_raw_10 \
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
    --input_tensors ecg.ecg_rest_median_raw_10 --output_tensors ecg.ecg_rest_median_raw_10 \
    --model_file /path/to/output/ecg_median_autoencoder/ecg_median_autoencoder.h5 \
    --id ecg_median_autoencoder 
```

With this latent space and phecode diagnoses for the same cohort, the jupyter notebook 
[latent_space_phewas](./latent_space_phewas.ipynb)
allows you to conduct the PheWAS analysis.

![UKB PheWAS Plot](./ukb_phewas.png)
