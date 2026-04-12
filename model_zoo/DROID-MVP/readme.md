# DROID-MVP Inference Example

This is a simple example script demonstrating how to load and run the DROID-MVP model. Model training and inference was performed using the code provided in the ML4H [model zoo](https://github.com/broadinstitute/ml4h/tree/master/model_zoo/DROID). The example below was adapted from the DROID inference code.

1. Build the DROID docker image from the official `ml4h` TensorFlow 2.19 image.

`docker build -f docker/droid/Dockerfile -t ml4h-droid:tf2.19 .`

2.  Pull github repo, including DROID-MVP model checkpoints stored using git lfs.

```
github clone https://github.com/broadinstitute/ml4h.git
git lfs pull --include ml4h/model_zoo/DROID-MVP/droid_mvp_checkpoint/*
git lfs pull --include ml4h/model_zoo/DROID-MVP/movinet_a2_base/*
```

3. Run the DROID image while mounting the `ml4h` directory and run the example inference script.

`docker run -it --rm -v {PATH TO CLONED ML4H DIRECTORY}:/ml4h/ ml4h-droid:tf2.19`

```
cd /ml4h/model_zoo/DROID-MVP/
python droid_mvp_inference.py
```

To use with your own data, format echocardiogram videos as tensors with shape (16, 224, 224, 3) before passing to the model. Code for data preprocessing, storage, loading, training, and inference can be found in the ml4h [model zoo](https://github.com/broadinstitute/ml4h/tree/master/model_zoo/DROID).

Model outputs for DROID-MVP take the form: 
```
[
    [["MVP", "Not MVP"]], 
    [["Anterior ", "Bileaflet", "Not MVP", "Posterior", "Superior Displacement", "MVP not otherwise specified"]], 
]
```

Note that the model was optimized for predicting binary MVP status (the primary task) and that detailed MVP status was used as an auxiliary task to improve performance on the primary classification task.
