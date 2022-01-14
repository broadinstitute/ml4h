# Estimating body fat distribution from silhouette images

## Description

This repo contains code to prepare silhouettes from UK Biobank whole-body magnetic resonance images and training deep learning models to estimate fat-depot volumes.

## Usage

Several files are provided:

* [ingest_mri.py](../../ml4h/applications/ingest/ingest_mri.py): ingesting UKB MRI data
* [two_d_projection.py](../../ml4h/applications/ingest/two_d_projection.py): computing 2-dimensional projections
* [ingest_autosegment.py](../../ml4h/applications/ingest/ingest_autosegment.py): autosegmenting axial slices
* [train_models.py](./train_models.py): training deep-learning models
* [callbacks.py](./callbacks.py): supporting callbacks required during training
* [shrinkage_loss.py](./shrinkage_loss.py): supportive loss required during training

### Citation

**Estimating body fat distribution - a driver of cardiometabolic health - from silhouette images**, Marcus D. R. Klarqvist, PhD*, Saaket Agrawal, BS*, Nathaniel Diamant, BS, Patrick T. Ellinor, MD, PhD, Anthony Philippakis, MD, PhD,  Kenney Ng, PhD, Puneet Batra, PhD, Amit V. Khera, MD
