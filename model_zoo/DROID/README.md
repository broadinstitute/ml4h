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

## Usage
### Preprocessing
The following scripts are designed to handle echo movies that have been processed and stored in Lightning Memory-
Mapped Database (lmdb) files. We create one lmdb per echo study in which the keys are the filenames of the dicoms and
the values are echo movies that have been anonymized, cropped, and converted to avis. See `echo_to_lmdb.py` for an
example.

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
    --lmdb_folder {LMDB_DIRECTORY_PATH} \
    --pretrained_ckpt_dir {SPECIALIZED_ENCODER_PATH} \
    --movinet_ckpt_dir {MoViNet-A2-Base_PATH} \
    --output_dir {WHERE_TO_STORE_PREDICTIONS}
```