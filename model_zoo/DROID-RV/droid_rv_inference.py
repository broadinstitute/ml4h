#!/usr/bin/env python
# coding: utf-8

import sys

import numpy as np
import pandas as pd
import tensorflow as tf
import logging
tf.get_logger().setLevel(logging.ERROR)

sys.path.append('../DROID')
from data_descriptions.echo import LmdbEchoStudyVideoDataDescription
from model_descriptions.echo import create_movinet_classifier, create_regressor_classifier

droid_rv_checkpoint = "droid_rv_checkpoint/chkp"
droid_rvef_checkpoint = "droid_rvef_checkpoint/chkp"
movinet_chkp_dir = 'movinet_a2_base/'
test_data_dir = 'test_data'
n_input_frames = 16

# Training-set (std, mean) used to scale each regression output, in model output
# order. Applied in reverse below to map predictions back to physical units.
DROID_RV_REGRESSION_SCALING = [
    ('Age', 15.51761856, 64.43979878),
    ('RVEDD', 6.88963822, 42.52320993),
]
DROID_RVEF_REGRESSION_SCALING = [
    ('RVEF', 8.658711, 53.40699),
    ('RV End-Diastolic Volume', 46.5734, 130.8913),
    ('RV End-Systolic Volume', 31.6643, 62.87321),
    ('Age', 22.99643, 47.18989),
]

def rescale_regression_head(model_output, scaling):
    for i, (_, std, mean) in enumerate(scaling):
        model_output[0][:, i] = model_output[0][:, i] * std + mean
    return model_output

movinet_model, backbone = create_movinet_classifier(
    n_input_frames=n_input_frames,
    batch_size=16,
    num_classes=600,
    checkpoint_dir=movinet_chkp_dir,
)

backbone_output = backbone.layers[-1].output[0]
flatten = tf.keras.layers.Flatten()(backbone_output)
encoder = tf.keras.Model(inputs=[backbone.input], outputs=[flatten])

droid_rv_func_args = {
    'input_shape': (n_input_frames, 224, 224, 3),
    'n_output_features': 2, # number of regression features
    'categories': {"RV_size":2, "RV_function":2, "Sex":2},
    'category_order': ["RV_size", "RV_function", "Sex"],
}

droid_rvef_func_args = {
    'input_shape': (n_input_frames, 224, 224, 3),
    'n_output_features': 4, # number of regression features
    'categories': {"Sex":2},
    'category_order': ["Sex"],
}

droid_rv_model = create_regressor_classifier(encoder, **droid_rv_func_args)
droid_rv_model.load_weights(droid_rv_checkpoint)

droid_rvef_model = create_regressor_classifier(encoder, **droid_rvef_func_args)
droid_rvef_model.load_weights(droid_rvef_checkpoint)

# Load one sample video from the checked-in test data (see readme.md) using the
# same loader the DROID recipes use.
wide_file = pd.read_parquet(f'{test_data_dir}/wide_file.parquet')
sample_id = wide_file['sample_id'].iloc[0]

input_dd = LmdbEchoStudyVideoDataDescription(f'{test_data_dir}/lmdb', 'image', nframes=n_input_frames)
video = input_dd.get_raw_data(sample_id)
video_batch = np.expand_dims(video, axis=0)

droid_rv_pred = droid_rv_model.predict(video_batch)
droid_rvef_pred = droid_rvef_model.predict(video_batch)

print(f"""

Sample: {sample_id}

DROID-RV Predictions:
{rescale_regression_head(droid_rv_pred, DROID_RV_REGRESSION_SCALING)}

DROID-RVEF Predictions:
{rescale_regression_head(droid_rvef_pred, DROID_RVEF_REGRESSION_SCALING)}

""")