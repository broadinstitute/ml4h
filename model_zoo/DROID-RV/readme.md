# DROID-RV Inference Example

This is a simple example script demonstrated how to load and run the DROID-RV and DROID-RVEF models. Model training and inference was performed using the code provided in the ML4H [model zoo](https://github.com/broadinstitute/ml4h/tree/master/model_zoo/DROID). The example below was adapted from the DROID inference code.

1. Download DROID docker image. Note: docker image is not compatible with Apple Silicon.

`docker pull alalusim/droid:latest`

2.  Pull github repo, including DROID-RV model checkpoints stored using git lfs.

```
github clone https://github.com/broadinstitute/ml4h.git
git lfs pull --include ml4h/model_zoo/DROID-RV/droid_rv_checkpoint/*
git lfs pull --include ml4h/model_zoo/DROID-RV/droid_rvef_checkpoint/*
git lfs pull --include ml4h/model_zoo/DROID-RV/movinet_a2_base/*
```

3. Run docker image while mounting ml4h directory and run example inference script.

`docker run -it -v {PATH TO CLONED ML4H DIRECTORY}:/ml4h/ alalusim/droid:latest`

```
cd /ml4h/model_zoo/DROID-RV/
python droid_rv_inference.py
```

To use with your own data, format echocardiogram videos as tensors with shape (16, 224, 224, 3) before passing to the model. Code for data preprocessing, storage, loading, training, and inference can be found in the ML4H [model zoo](https://github.com/broadinstitute/ml4h/tree/master/model_zoo/DROID).

Model outputs for DROID-RV take the form: 
```
[
    "Age", 
    "RVEDD", 
    {"RV Size": {"Dilated", "Not Dilated"}, 
    "RV Systolic Function": {"Hypokinetic", "Not Hypokinetic"}, 
    "Sex": {"Female", "Male"}
]
```

Model outputs for DROID-RVEF take the form: 
```
[
    "RVEF", 
    "RV End-Diastolic Volume,
    "RV End-Systolic Volume",
    "Age",
    "Sex": {"Female", "Male"}
]
```