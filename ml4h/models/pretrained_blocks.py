import logging
from typing import Dict, List, Tuple, Sequence

import numpy as np
import tensorflow as tf
import keras
import tensorflow_hub as hub
from official.projects.movinet.modeling import movinet
from official.projects.movinet.modeling import movinet_model
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

    def __call__(self, x: Tensor, intermediates: Dict[TensorMap, List[Tensor]] = None) -> Tensor:
        if not self.can_apply():
            return x
        x = self.base_model(x, training=False)
        x = keras.layers.GlobalAveragePooling2D()(x)
        intermediates[self.tensor_map].append(x)
        return x

import tensorflow as tf
from tensorflow.keras import layers, Model

def conv2plus1d(x, filters, kernel_size=(3, 3, 3), strides=(1, 1, 1), name=None):
    t, h, w = kernel_size
    st, sh, sw = strides

    x = layers.Conv3D(
        filters,
        kernel_size=(1, h, w),
        strides=(1, sh, sw),
        padding="same",
        use_bias=False,
        name=None if name is None else f"{name}_spatial",
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.Conv3D(
        filters,
        kernel_size=(t, 1, 1),
        strides=(st, 1, 1),
        padding="same",
        use_bias=False,
        name=None if name is None else f"{name}_temporal",
    )(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    return x

def residual_block(x, filters, downsample=False, name=None):
    stride = (1, 2, 2) if downsample else (1, 1, 1)
    shortcut = x

    y = conv2plus1d(x, filters, strides=stride, name=None if name is None else f"{name}_conv1")
    y = conv2plus1d(y, filters, name=None if name is None else f"{name}_conv2")

    if downsample or x.shape[-1] != filters:
        shortcut = layers.Conv3D(
            filters,
            kernel_size=1,
            strides=stride,
            padding="same",
            use_bias=False,
            name=None if name is None else f"{name}_proj",
        )(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)

    y = layers.Add()([shortcut, y])
    y = layers.Activation("relu")(y)
    return y

def build_movinet_style_encoder(
    input_shape=(16, 224, 224, 3),
    embedding_dim=512,
):
    inputs = layers.Input(shape=input_shape)

    x = conv2plus1d(inputs, 32, kernel_size=(3, 7, 7), strides=(1, 2, 2), name="stem")
    x = residual_block(x, 32, name="res1")
    x = residual_block(x, 64, downsample=True, name="res2")
    x = residual_block(x, 64, name="res3")
    x = residual_block(x, 128, downsample=True, name="res4")
    x = residual_block(x, 128, name="res5")
    x = residual_block(x, 256, downsample=True, name="res6")

    x = layers.GlobalAveragePooling3D(name="gap")(x)
    x = layers.Dense(embedding_dim, activation=None, name="embedding")(x)

    return Model(inputs, x, name="movinet_style_encoder")


class MoviNetEncoder(Block):
    def __init__(self, tensor_map, dense_layers, pretrain_trainable, **kwargs):
        self.tensor_map = tensor_map
        if self.tensor_map.axes() != 4:
            return

        self.base_model = build_movinet_style_encoder(
            input_shape=(16, 224, 224, 3),
            embedding_dim=dense_layers[-1],
        )
        self.base_model.trainable = pretrain_trainable

    def can_apply(self):
        return self.tensor_map.axes() == 4

    def __call__(self, x, intermediates=None):
        if not self.can_apply():
            return x
        encoding = self.base_model(x, training=False)
        intermediates[self.tensor_map].append(encoding)
        return encoding


class BertEncoder(Block):
    def __init__(
            self,
            *,
            tensor_map: TensorMap,
            pretrain_trainable: bool,
            base_model="https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/3",
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

    def __call__(self, x: Tensor, intermediates: Dict[TensorMap, List[Tensor]] = None) -> Tensor:
        encoder_inputs = self.preprocess_model(x)
        outputs = self.encoder(encoder_inputs)
        intermediates[self.tensor_map].append(outputs['pooled_output'])
        return outputs['pooled_output']
