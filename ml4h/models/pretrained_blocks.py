import logging
from typing import Dict, List, Tuple, Sequence

import numpy as np
import tensorflow as tf
from tensorflow import keras
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
            **kwargs,
    ):
        self.tensor_map = tensor_map
        if not self.can_apply():
            return
        self.base_model = keras.applications.Xception(
            weights="imagenet",  # Load weights pre-trained on ImageNet.
            input_shape=self.tensor_map.shape,
            include_top=False,
        )  # Do not include the ImageNet classifier at the top.
        self.base_model.trainable = False

    def can_apply(self):
        return self.tensor_map.axes() > 1

    def __call__(self, x: Tensor, intermediates: Dict[TensorMap, List[Tensor]]) -> Tensor:
        if not self.can_apply():
            return x
        x = self.base_model(x, training=False)
        x = keras.layers.GlobalAveragePooling2D()(x)
        intermediates[self.tensor_map].append(x)
        return x