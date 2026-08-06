# DROID-RV Overview

## Software Requirements

DROID-RV was trained and tested using Python 3.6.9 with packages detailed in [requirements.txt](requirements.txt). A Docker image containing all necessary software to run model training and inference can be found on [Docker Hub](https://hub.docker.com/r/alalusim/droid). The model was trained and tested on x86 CPUs using Nvidia v100 GPUs.

## Installation


1. Download DROID docker image from Docker Hub. Download time will vary depending on connection speed but should be <20 minutes. Note: docker image is not compatible with Apple Silicon.

`docker pull alalusim/droid:latest`

2.  Pull github repo, including DROID-RV model checkpoints and synthetic test data stored using git lfs. Download time should be <5 minutes and will vary with connection speed.

```
git clone https://github.com/broadinstitute/ml4h.git
cd ml4h
git lfs pull --include model_zoo/DROID-RV/droid_rv_checkpoint/*
git lfs pull --include model_zoo/DROID-RV/droid_rvef_checkpoint/*
git lfs pull --include model_zoo/DROID-RV/movinet_a2_base/*
git lfs pull --include model_zoo/DROID-RV/test_data/*
```

## Inference Example

This is a simple example script demonstrating how to load and run the DROID-RV and DROID-RVEF models. Model training and inference was performed using the code provided in the ML4H [model zoo](https://github.com/broadinstitute/ml4h/tree/master/model_zoo/DROID). The example below was adapted from the DROID inference code. It loads a sample video from `test_data/` (containing random noise) and prints predictions from both DROID-RV and DROID-RVEF checkpoints. Inference takes <1 minute per sample to run on a CPU.

Run docker image while mounting ml4h directory and run example inference script.

`docker run -it -v {PATH TO CLONED ML4H DIRECTORY}:/ml4h/ alalusim/droid:latest`

```
cd /ml4h/model_zoo/DROID-RV/
python droid_rv_inference.py
```

To use with your own data, format echocardiogram videos as tensors with shape (16, 224, 224, 3) before passing to the model. Code for data preprocessing, storage, loading, training, and inference can be found in the ML4H [model zoo](https://github.com/broadinstitute/ml4h/tree/master/model_zoo/DROID).

Model outputs for DROID-RV take the form: 
```
[
    [["Age", "RVEDD"]], 
    [["Dilated", "Not Dilated"]], 
    [["Hypokinetic", "Not Hypokinetic"]], 
    [["Female", "Male"]]
]
```

Model outputs for DROID-RVEF take the form: 
```
[
    [["RVEF", "RV End-Diastolic Volume, "RV End-Systolic Volume", "Age"]],
    [["Female", "Male"]]
]
```

Expected output from the inference test script is:

```
Sample: 170682_170682_0fe9f189

DROID-RV Predictions:
[array([[62.34924 , 43.112488]], dtype=float32), array([[0.5758543 , 0.42414567]], dtype=float32), array([[0.5898055 , 0.41019455]], dtype=float32), array([[0.51277626, 0.4872237 ]], dtype=float32)]

DROID-RVEF Predictions:
[array([[ 52.2796  , 137.59207 ,  66.795494,  44.353607]], dtype=float32), array([[0.49082455, 0.5091754 ]], dtype=float32)]
```

## Training Example

Data preprocessing and model training was performed using the DROID training recipe as seen in the [DROID model zoo entry](https://github.com/broadinstitute/ml4h/tree/master/model_zoo/DROID). The Docker image described above can also be used for model training.

Requirements:
- Wide file: Parquet file with one row per TTE video, and columns corresponding to sample identifier, patient split, and outcome(s) of interest
- Splits file: JSON file with keys corresponding to "patient_train", "patient_valid", "patient_test" and values corresponding to lists of patient identifiers
- LMDB folder: prepared as described in the [DROID model zoo entry](https://github.com/broadinstitute/ml4h/tree/master/model_zoo/DROID)
- Movinet checkpoint: movinet_a2_base in the [DROID-RV model zoo entry](https://github.com/broadinstitute/ml4h/tree/master/model_zoo/DROID-RV)
- (Optional) pretrained checkpoint: for example https://github.com/broadinstitute/ml4h/tree/master/model_zoo/DROID-RV/droid_rv_checkpoint

For valid arguments for view, Doppler, quality, and canonical axis, see model_zoo/DROID/echo_defines.py.

The command below is an example of how the training script can be used:

```
python model_zoo/DROID/echo_supervised_training_recipe.py \
    --n_input_frames 16 \
    --output_labels age \ # output labels must be present as a column in the wide file
    --output_labels sex \ 
    --output_labels rvedd \
    --output_labels rv_size \
    --output_labels rv_function \
    --output_labels_types rcrcc \ # r = regression or c = classification; must be given in same the order as output labels above
    --wide_file {WIDE_FILE_PATH} \
    --lmdb_folder {LMDB_FOLDER_PATH} \
    --splits_file {SPLITS_JSON_FILE_PATH} \
    --selected_views A4C \ # see model_zoo/DROID/echo_defines.py for view/Doppler/quality/canonical axis arguments
    --selected_views RV_focused \ 
    --selected_doppler standard \
    --selected_quality good \
    --selected_canonical on_axis \
    --n_train_patients all \
    --batch_size 16\
    --epochs 50 \
    --es_patience 5 \
    --scale_outputs \
    --skip_modulo 4\
    --adam 1e-4 \
    --movinet_chkp_dir {MOVINET_CHECKPOINT_PATH} \ # corresponds to ml4h/model_zoo/DROID-RV/movinet_a2_base/chkp
    --pretrained_chkp_dir {PRETRAINED_CHECKPOINT_PATH} \ # used when fine-tuning, for example can provide ml4h/model_zoo/DROID-RV/droid_rv_checkpoint/chkp
    --output_dir {OUTPUT_FOLDER_PATH}
```

A ready-to-run version of this command using synthetic test data is provided in `run_training_example.sh`. Running one epoch against the test data takes approximately *** minutes on a CPU.

```
cd /ml4h/model_zoo/DROID-RV/
./run_training_example.sh
```

The test script will output a model checkpoint to `../DROID-RV/training_example_output/` and takes approximately 1 hour to run on a CPU.
