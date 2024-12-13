#!/usr/bin/env python
# coding: utf-8

import numpy as np
import tensorflow as tf
from droid_mvp_model_description import create_movinet_classifier, create_regressor_classifier
import logging 
tf.get_logger().setLevel(logging.ERROR)

pretrained_chkp_dir = "droid_mvp_checkpoint/chkp"
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

func_args = {
    'input_shape': (16, 224, 224, 3),
    'n_output_features': 0, # number of regression features
    'categories': {"mvp_status_binary":2, "mvp_status_detailed":6},
    'category_order': ["mvp_status_binary", "mvp_status_detailed"],
}

model_plus_head = create_regressor_classifier(encoder, **func_args)

model_plus_head.load_weights(pretrained_chkp_dir)

random_video = np.random.random((1, 16, 224, 224, 3))
print(model_plus_head.predict(random_video))
