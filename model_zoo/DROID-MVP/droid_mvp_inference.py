#!/usr/bin/env python
# coding: utf-8

import numpy as np
import tensorflow as tf
#from droid_mvp_model_description import create_movinet_classifier, create_regressor_classifier
from droid_mvp_model_description import create_regressor_classifier
import logging 
tf.get_logger().setLevel(logging.ERROR)

pretrained_chkp_dir = "/work/data/alalusim/echo_mvp/mvp_trained_models/202407111151_PLAX_A4C_A3C_A2C_mvp_label_1_1_MV_Prolapse_16frames_all/model/chkp"
# movinet_chkp_dir = '/work/data/movinet_a2_base/'
# 
# movinet_model, backbone = create_movinet_classifier(
#     n_input_frames=16,
#     batch_size=16,
#     num_classes=600,
#     checkpoint_dir=movinet_chkp_dir,
# )
# 
# backbone_output = backbone.layers[-1].output[0]
# flatten = tf.keras.layers.Flatten()(backbone_output)
# encoder = tf.keras.Model(inputs=[backbone.input], outputs=[flatten])
import tensorflow_hub as hub

hub_url = "https://www.kaggle.com/models/google/movinet/TensorFlow2/a2-base-kinetics-600-classification/3"

encoder = hub.KerasLayer(hub_url, trainable=False)

inputs = tf.keras.layers.Input(
    shape=[None, None, None, 3],
    dtype=tf.float32,
    name='image')

# [batch_size, 600]
outputs = encoder(dict(image=inputs))

model = tf.keras.Model(inputs, outputs, name='movinet')

func_args = {
    'input_shape': (16, 224, 224, 3),
    'n_output_features': 0, # number of regression features
    'categories': {"mvp_status_binary":2, "mvp_status_detailed":6},
    'category_order': ["mvp_status_binary", "mvp_status_detailed"],
}

model_plus_head = create_regressor_classifier(model, **func_args)

model_plus_head.load_weights(pretrained_chkp_dir)

random_video = np.random.random((1, 16, 224, 224, 3))
model_plus_head(random_video)