import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras.models import load_model
from tensorflow_addons.optimizers import RectifiedAdam
import numpy as np

import ml4h
from ml4h.metrics import get_metric_dict
import ml4h.tensormap.ukb.mri

# Path to a pre-trained file
model_file = "models_sax_segment_jamesp_dropout_200_sax_segment_jamesp_dropout_200.h5"
# Get the config using the TensorMap used to train the model in the first place.
# This allows us to reconstruct models that are not saved using configs.
objects = get_metric_dict([ml4h.tensormap.ukb.mri.cine_segmented_sax_slice_jamesp])
# Silly work-around to reset function pointer to our local instance.
objects['RectifiedAdam'] = RectifiedAdam
# Load the model
model = load_model(model_file, custom_objects=objects)
model.summary()
# Example output ...
# __________________________________________________________________________________________________
# output_cine_segmented_sax_slice (None, 224, 224, 12) 1164        concatenate_33[0][0]
# ==================================================================================================
# Total params: 5,404,204
# Trainable params: 5,404,204
# Non-trainable params: 0
# __________________________________________________________________________________________________


import ml4h.evaluate_segmentations as evaluate
# from ml4h.models import make_multimodal_multitask_model
# from ml4h.TensorMap import TensorMap, Interpretation, decompress_data
from ml4h.tensor_generators import TensorGenerator, test_train_valid_tensor_generators
import glob
import os

files = glob.glob('data/*.hd5') * 10 # Repeat the same test file 10 times

generate_test = TensorGenerator(
    1, 
    [ml4h.tensormap.ukb.mri.sax_slice_jamesp], 
    [ml4h.tensormap.ukb.mri.cine_segmented_sax_slice_jamesp], 
    paths=files,
    num_workers=1,
    cache_size=0, 
    keep_paths=True, 
    mixup=0,
)


dg = evaluate.DataGenerator(files, generate_test)
eval, eval_batch, prediction_prob, predictions = evaluate.evaluate_collect_metrics(model, dg)

evaluations = evaluate.evaluate_segmentation_models([model, model], [generate_test])