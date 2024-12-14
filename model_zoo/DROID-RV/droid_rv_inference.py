#!/usr/bin/env python
# coding: utf-8

import numpy as np
import tensorflow as tf
from droid_rv_model_description import create_movinet_classifier, create_regressor_classifier
import logging 
tf.get_logger().setLevel(logging.ERROR)

droid_rv_checkpoint = "droid_rv_checkpoint/chkp"
droid_rvef_checkpoint = "droid_rvef_checkpoint/chkp"
movinet_chkp_dir = 'movinet_a2_base/'

movinet_model, backbone = create_movinet_classifier(
    n_input_frames=16,
    batch_size=16,
    num_classes=600,
    checkpoint_dir=movinet_chkp_dir,
)

backbone_output = backbone.layers[-1].output[0]
flatten = tf.keras.layers.Flatten()(backbone_output)
encoder = tf.keras.Model(inputs=[backbone.input], outputs=[flatten])

droid_rv_func_args = {
    'input_shape': (16, 224, 224, 3),
    'n_output_features': 2, # number of regression features
    'categories': {"RV_size":2, "RV_function":2, "Sex":2},
    'category_order': ["RV_size", "RV_function", "Sex"],
}

droid_rvef_func_args = {
    'input_shape': (16, 224, 224, 3),
    'n_output_features': 0, # number of regression features
    'categories': {"RV_size":2, "RV_function":2, "Sex":2},
    'category_order': ["RV_size", "RV_function", "Sex"],
}

droid_rv_model = create_regressor_classifier(encoder, **droid_rv_func_args)
droid_rv_model.load_weights(droid_rv_checkpoint)

droid_rvef_model = create_regressor_classifier(encoder, **droid_rvef_func_args)
droid_rvef_model.load_weights(droid_rvef_checkpoint)

random_video = np.random.random((1, 16, 224, 224, 3))

print(f"""

DROID-RV Predictions:
{droid_rv_model.predict(random_video)}

DROID-RVEF Predictions:
{droid_rvef_model.predict(random_video)}

""")