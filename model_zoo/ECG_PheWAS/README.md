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

Our models expects ECG median waveforms with 600 voltages across 12 leads as input with which it produces 
a 256 dimensional latent space encoding, and a reconstructed ECG with the same shape as the input.
