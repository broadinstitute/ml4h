# DROID-RV Overview

## Inference Example

This is a simple example script demonstrating how to load and run the DROID-RV and DROID-RVEF models. Model training and inference was performed using the code provided in the ML4H [model zoo](https://github.com/broadinstitute/ml4h/tree/master/model_zoo/DROID). The example below was adapted from the DROID inference code.

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