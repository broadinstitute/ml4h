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