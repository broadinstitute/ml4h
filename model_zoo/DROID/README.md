# DROID (Dimensional Reconstruction of Imaging Data)

DROID is a 3-D convolutional neural network modeling approach for echocardiographic view
classification and quantification of LA dimension, LV wall thickness, chamber diameter and
ejection fraction.

The DROID echo movie encoder is based on the 
[MoViNet-A2-Base](https://tfhub.dev/tensorflow/movinet/a2/base/kinetics-600/classification/3) 
video classification model. MoViNet was fine-tuned in a supervised fashion to produce two
specialized encoders:
- DROID-LA
  - input views: PLAX, A4C, A2C
  - output predictions: LA A/P
- DROID-LV
  - input views: PLAX, A4C, A2C
  - output predictions: LVEF, LVEDD, LVESD, IVS, PWT

Multi-instance attention heads were then trained to integrate up to 40 view encodings to predict
a single measurement of each type per echo study.

## Requirements
In addition to the `ml4h` repository, DROID also requires `ml4ht_data_source` plus other dependencies. First, clone the
ml4h repositories:
```commandline
git clone https://github.com/broadinstitute/ml4h.git
git clone https://github.com/broadinstitute/ml4ht_data_source.git
```

For convenience, we provide a docker image containing additional dependencies:
```commandline
docker run -it --gpus all --rm -v {PARENT_DIRECTORY_OF_REPOS} -v {OPTIONAL_DATA_DIRECTORY} \
us-central1-docker.pkg.dev/broad-ml4cvd/droid/droid:0.1 /bin/bash
```

Within the docker container, install `ml4ht`:
```commandline
pip install --user ml4ht_data_source
```

## Usage

For parallel video loading, synthetic benchmarks, and V100 tuning, see
[VIDEO_LOADING.md](VIDEO_LOADING.md). The training recipe defaults to four clip
workers, selective MJPEG decoding, and one prefetched batch; the LMDB format is unchanged.

### Preprocessing
The following scripts are designed to handle echo movies that have been processed and stored in Lightning 
Memory-Mapped Database (lmdb) files. We create one lmdb per echo study in which the keys are the filenames of the dicoms and
the values are echo movies that have been anonymized, cropped, and converted to avis. See `echo_to_lmdb.py` for an
example.

### Survival training

`echo_supervised_training_recipe.py` supports one discrete-time survival task alongside ordinary regression and classification outputs. Supply the task name and each required survival argument:

```commandline
python echo_supervised_training_recipe.py \
    --survival_task incident_hf \
    --survival_event_column hf_event \
    --survival_follow_up_days_column hf_follow_up_days \
    --survival_intervals 25 \
    --survival_days_window 3650 \
    ...standard training arguments...
```

The model predicts one conditional survival probability per interval. The recipe encodes the label internally as survival indicators plus an event-bin indicator and treats an event at or before the index time as a first-interval event by default. Add `--survival_prevalent_policy exclude` to remove those cases, or `--survival_blanking_days 30` to apply a 30-day blanking period. If `--survival_task` is present, the recipe raises an error unless every required survival argument is supplied.

### Inference
`echo_supervised_inference_recipe.py` can be used to obtain predictions from echo movies given either the DROID-LA or
DROID-LV specialized encoders.

An example of parameters to use when running this script are:
```commandline
python echo_supervised_inference_recipe.py \
    --n_input_frames 16 \
    --output_labels LA_A_P \
    --selected_views A4C --selected_views A2C --selected_views PLAX \
    --selected_doppler standard \
    --selected_quality good \
    --selected_canonical on_axis \
    --split_idx 0 \
    --n_splits 1 \
    --skip_modulo 4 \
    --wide_file {WIDE_FILE_PATH} \
    --splits_file {SPLITS_JSON} \
    --lmdb_folder {LMDB_DIRECTORY_PATH} \
    --pretrained_chkp_dir {SPECIALIZED_ENCODER_PATH} \
    --movinet_chkp_dir {MoViNet-A2-Base_PATH} \
    --output_dir {WHERE_TO_STORE_PREDICTIONS}
```

The heads (regression, classification and survival) are rebuilt from the run directory that contains
`--pretrained_chkp_dir` (`model_params.json` and, for classification, the class mapping), using
`model_descriptions/droid_model.py`, the same builder the training recipe uses. Views, quality filters,
`--n_input_frames` and `--skip_modulo` default to the checkpoint's training values. Clips load through the
parallel loader and accept the training recipe's `--loader_workers`, `--video_decode_mode`,
`--video_decode_threads` and `--prefetch_batches` options.

`prediction_{split_idx}.pq` holds `prediction_i` regression columns, one class-label column per
classification head (per-class probabilities are in `prediction_{split_idx}_one_hot_<label>.pq`), and
`survival_<task>_i` per-interval conditional survival probabilities with their product,
`survival_<task>_cumulative`. `--extract_embeddings` also saves the fine-tuned encoder's `embedding_i`
columns, one row per clip, to the matching `inference_embeddings_*` folder, computed in the same pass as the
predictions.
