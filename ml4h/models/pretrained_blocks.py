import logging
from typing import Dict, List, Tuple, Sequence

import numpy as np
import tensorflow as tf
from tensorflow import keras
import tensorflow_hub as hub
from tensorflow.keras.layers import Dense, Flatten, Reshape, LayerNormalization, DepthwiseConv2D, concatenate, Concatenate, Add

from ml4h.models.Block import Block
from ml4h.TensorMap import TensorMap
from ml4h.models.basic_blocks import DenseBlock
from ml4h.models.layer_wrappers import _upsampler, _activation_layer, _regularization_layer, _normalization_layer
from ml4h.models.layer_wrappers import _conv_layer_from_kind_and_dimension, _pool_layers_from_kind_and_dimension, _one_by_n_kernel

Tensor = tf.Tensor


class ResNetEncoder(Block):
    def __init__(
            self,
            *,
            tensor_map: TensorMap,
            pretrain_trainable: bool,
            **kwargs,
    ):
        self.tensor_map = tensor_map
        if not self.can_apply():
            return
        self.base_model = keras.applications.ResNet50V2(
            input_shape=self.tensor_map.shape,
            weights="imagenet",  # Load weights pre-trained on ImageNet.
            #pooling = "avg",
            include_top=False,
        )  # Do not include the ImageNet classifier at the top.
        self.base_model.trainable = pretrain_trainable

    def can_apply(self):
        return self.tensor_map.axes() == 3 and self.tensor_map.shape[-1] == 3

    def __call__(self, x: Tensor, intermediates: Dict[TensorMap, List[Tensor]]) -> Tensor:
        if not self.can_apply():
            return x
        x = self.base_model(x, training=False)
        x = keras.layers.GlobalAveragePooling2D()(x)
        intermediates[self.tensor_map].append(x)
        return x


class ConvNeXtBaseEncoder(Block):
    def __init__(
            self,
            *,
            tensor_map: TensorMap,
            pretrain_trainable: bool,
            **kwargs,
    ):
        self.tensor_map = tensor_map
        if not self.can_apply():
            return
        self.base_model = keras.applications.EfficientNetB0(
            input_shape=self.tensor_map.shape,
            weights="imagenet",  # Load weights pre-trained on ImageNet.
            #pooling = "avg",
            include_top=False,
        )  # Do not include the ImageNet classifier at the top.
        self.base_model.trainable = pretrain_trainable

    def can_apply(self):
        return self.tensor_map.axes() == 3 and self.tensor_map.shape[-1] == 3

    def __call__(self, x: Tensor, intermediates: Dict[TensorMap, List[Tensor]]) -> Tensor:
        if not self.can_apply():
            return x
        x = self.base_model(x, training=False)
        x = keras.layers.GlobalAveragePooling2D()(x)
        intermediates[self.tensor_map].append(x)
        return x


class MoviNetEncoder(Block):
    def __init__(
            self,
            *,
            tensor_map: TensorMap,
            pretrain_trainable: bool,
            path='https://tfhub.dev/tensorflow/movinet/a2/base/kinetics-600/classification/3',
            **kwargs,
    ):
        self.tensor_map = tensor_map
        if not self.can_apply():
            return
        self.base_model = hub.KerasLayer(path, trainable=pretrain_trainable)
        self.base_model.trainable = pretrain_trainable

    def can_apply(self):
        return self.tensor_map.axes() == 4

    def __call__(self, x: Tensor, intermediates: Dict[TensorMap, List[Tensor]]) -> Tensor:
        if not self.can_apply():
            return x
        x = self.base_model({'image': x}, training=True)
        #x = keras.layers.GlobalAveragePooling2D()(x)
        intermediates[self.tensor_map].append(x)
        return x


class BertEncoder(Block):
    def __init__(
            self,
            *,
            tensor_map: TensorMap,
            pretrain_trainable: bool,
            base_model = "https://tfhub.dev/google/experts/bert/wiki_books/sst2/2",
            preprocess_model="https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3",
            **kwargs,
    ):
        self.tensor_map = tensor_map
        if not self.can_apply():
            return
        self.preprocess_model = hub.KerasLayer(preprocess_model, name='preprocessing')
        self.encoder = hub.KerasLayer(base_model, trainable=pretrain_trainable, name='BERT_encoder')

    def can_apply(self):
        return self.tensor_map.is_text()

    def __call__(self, x: Tensor, intermediates: Dict[TensorMap, List[Tensor]]) -> Tensor:
        encoder_inputs = self.preprocess_model(x)
        outputs = self.encoder(encoder_inputs)
        intermediates[self.tensor_map].append(outputs['pooled_output'])
        return outputs['pooled_output']

#
#
# def build_classifier_model(tfhub_handle_preprocess, tfhub_handle_encoder, tensor_maps_out):
#     text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')
#     preprocessing_layer = hub.KerasLayer(tfhub_handle_preprocess, name='preprocessing')
#     encoder_inputs = preprocessing_layer(text_input)
#     encoder = hub.KerasLayer(tfhub_handle_encoder, trainable=True, name='BERT_encoder')
#     outputs = encoder(encoder_inputs)
#     net = outputs['pooled_output']
#     net = tf.keras.layers.Dropout(dropout_rate)(net)
#     #net = tf.keras.layers.Dense(256, activation='swish')(net)
#     #net = tf.keras.layers.Dropout(dropout_rate)(net)
#     outputs = []
#     metrics = []
#     losses = []
#     for otm in tensor_maps_out:
#         if otm.is_categorical():
#             outputs.append(tf.keras.layers.Dense(len(otm.channel_map), activation=None, name=otm.name)(net))
#             #losses.append(tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))
#             #metrics.append(tf.metrics.SparseCategoricalAccuracy(name=f'{otm.name}_SparseCategoricalAccuracy_met'))
#         elif otm.is_continuous():
#             l1 = tf.keras.layers.Dense(64, activation='swish')(net)
#             l1 = tf.keras.layers.Dropout(dropout_rate)(l1)
#             outputs.append(tf.keras.layers.Dense(1, activation=None, name=otm.name)(l1))
#             #losses.append(tf.keras.losses.MeanSquaredError())
#             #metrics.append(tf.metrics.MeanAbsoluteError(name=f'{otm.name}_mae'))
#     return tf.keras.Model(text_input, outputs), losses, metrics
#
