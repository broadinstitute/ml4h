import logging
from typing import Dict, List, Tuple, Sequence

import numpy as np
import tensorflow as tf
import keras
import tensorflow_hub as hub
from official.projects.movinet.modeling import movinet
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
        self.base_model = movinet.Movinet(model_id='a2')
        self.base_model.build([None, 16, 224, 224, 3])
        dummy_input = tf.random.uniform([1, 16, 224, 224, 3])
        backbone_output = self.base_model(dummy_input)

        #self.base_model.trainable = pretrain_trainable

    def can_apply(self):
        return self.tensor_map.axes() == 4

    def __call__(self, x: Tensor, intermediates: Dict[TensorMap, List[Tensor]] = None) -> Tensor:
        if not self.can_apply():
            return x
        print(f'X is {x}')
        y = self.base_model(x)
        encoding = tf.keras.layers.Flatten()(y[0]['head'])
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
