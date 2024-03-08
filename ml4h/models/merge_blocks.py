import logging
from typing import Dict, List, Tuple

import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
import tensorflow_probability as tfp
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.layers import concatenate, Flatten, Average, Layer, Lambda, Dense

from ml4h.models.Block import Block
from ml4h.TensorMap import TensorMap
from ml4h.models.basic_blocks import DenseBlock
from ml4h.models.layer_wrappers import global_average_pool
from tensorflow.keras.losses import categorical_crossentropy


Tensor = tf.Tensor
tfd = tfp.distributions


class FlatConcatBlock(Block):
    """
    Flattens then concatenates all inputs
    """
    def __init__(self,  **kwargs):
        pass

    def __call__(self, x: Tensor, intermediates: Dict[TensorMap, List[Tensor]] = None) -> Tensor:
        y = [Flatten()(x[-1]) for tm, x in intermediates.items() if not tm.is_embedding()]
        y = concatenate(y) if len(y) > 1 else y[0]
        return y


class FlatConcatDenseBlock(Block):
    """
    Flattens then concatenates all inputs, applies a dense layer
    """
    def __init__(
            self,
            activation: str = 'swish',
            dense_layers: List[int] = [32],
            dense_normalize: str = None,
            dense_regularize: str = None,
            dense_regularize_rate: float = 0.0,
            **kwargs,
    ):
        self.fully_connected = DenseBlock(
            widths=dense_layers,
            activation=activation,
            normalization=dense_normalize,
            regularization=dense_regularize,
            regularization_rate=dense_regularize_rate,
            name='embed',
        ) if dense_layers else None

    def __call__(self, x: Tensor, intermediates: Dict[TensorMap, List[Tensor]] = None) -> Tensor:
        y = [Flatten()(x[-1]) for tm, x in intermediates.items()]
        y = concatenate(y) if len(y) > 1 else y[0]
        y = self.fully_connected(y, intermediates) if self.fully_connected else y
        return y


class GlobalAveragePoolBlock(Block):
    """
    GAPs then concatenates all inputs, applies a dense layer
    """
    def __init__(
            self,
            activation: str = 'swish',
            dense_layers: List[int] = [32],
            dense_normalize: str = None,
            dense_regularize: str = None,
            dense_regularize_rate: float = 0.0,
            **kwargs,
    ):
        self.fully_connected = DenseBlock(
            widths=dense_layers,
            activation=activation,
            normalization=dense_normalize,
            regularization=dense_regularize,
            regularization_rate=dense_regularize_rate,
            name='embed',
        ) if dense_layers else None

    def __call__(self, x: Tensor, intermediates: Dict[TensorMap, List[Tensor]] = None) -> Tensor:
        y = [Flatten()(x[-1]) for tm, x in intermediates.items() if tm.axes() == 1]  # Flat tensors
        y += [global_average_pool(x[-2 if self.fully_connected else -1]) for tm, x in intermediates.items() if tm.axes() > 1]  # Structured tensors
        y = concatenate(y) if len(y) > 1 else y[0]
        y = self.fully_connected(y, intermediates) if self.fully_connected else y
        return y


class AverageBlock(Block):
    """
    Average the last tensors in intermediates dictionary
    """
    def __init__(self, **kwargs):
        pass

    def __call__(self, x: Tensor, intermediates: Dict[TensorMap, List[Tensor]] = None) -> Tensor:
        return Average()([t[-1] for tm, t in intermediates.items()])


class ReduceMean(Block):
    """
    Average the last tensors in intermediates dictionary
    """
    def __init__(self, **kwargs):
        pass

    def __call__(self, x: Tensor, intermediates: Dict[TensorMap, List[Tensor]] = None) -> Tensor:
        y = [x[-1] for tm, x in intermediates.items()]
        y = tf.math.reduce_mean(y, axis=0)
        return y


class EncodeIdentityBlock(Block):
    """
    Adds the input tensor to the intermediates dictionary, useful for TensorMaps with pretrained embeddings
    """
    def __init__(self, **kwargs):
        super()

    def __call__(self, x: Tensor, intermediates: Dict[TensorMap, List[Tensor]] = None) -> Tensor:
        return x


class PairLossBlock(Block):
    """
    Flattens or GAPs then concatenates all inputs, applies a dense layer, then restructures to provided shapes
    """
    def __init__(
            self,
            pairs: List[Tuple[TensorMap, TensorMap]],
            pair_loss: str = 'cosine',
            pair_loss_weight: float = 1.0,
            pair_merge: str = 'dropout',
            batch_size: int = 4,
            dense_layers: List[int] = [32],
            **kwargs,
    ):
        self.pairs = pairs
        self.pair_merge = pair_merge
        self.batch_size = batch_size
        self.encoding_size = dense_layers[-1]
        if pair_loss == 'cosine':
            self.loss_layer = CosineLossLayer(pair_loss_weight)
        elif pair_loss == 'euclid':
            self.loss_layer = L2LossLayer(pair_loss_weight)
        elif pair_loss == 'contrastive':
            self.loss_layer = ContrastiveLossLayer(pair_loss_weight, batch_size)
        else:
            raise ValueError(f'Unknown pair loss type: {pair_loss}')

    def __call__(self, x: Tensor, intermediates: Dict[TensorMap, List[Tensor]] = None) -> Tensor:
        y = []
        for left, right in self.pairs:
            y.extend(self.loss_layer([intermediates[left][-1], intermediates[right][-1]]))
        if self.pair_merge == 'average':
            return Average()(y)
        elif self.pair_merge == 'concat':
            return concatenate(y)
        elif self.pair_merge == 'dropout':
            random_index = tf.random.uniform(shape=[intermediates[left][-1].shape[-1]], maxval=len(y), dtype=tf.int32)
            ranger = tf.range(intermediates[left][-1].shape[-1])
            indices = tf.stack([random_index, ranger], axis=-1)
            tf_y = tf.convert_to_tensor(y)
            tf_y = tf.transpose(tf_y, perm=[0, 2, 1])
            tf_g = tf.gather_nd(tf_y, indices)
            out = tf.transpose(tf_g)
            return out
        elif self.pair_merge == 'kronecker':
            krons = []
            losses = []
            for left, right in self.pairs:
                eshape = tf.shape(intermediates[left][-1])
                kron_layer = Lambda(lambda tensors: tf.einsum('...i,...j->...ij', tensors[0], tensors[1]))
                y = self.loss_layer([intermediates[left][-1], intermediates[right][-1]])
                kron = kron_layer(y)
                krons.append(tf.reshape(kron, [eshape[0], self.encoding_size*self.encoding_size]))
            if len(self.pairs) > 1:
                kron = concatenate(krons)
            else:
                kron = krons[0]
            kron = Dense(self.encoding_size)(kron)
            return kron
        else:
            raise ValueError(f'Unknown pair merge method: {self.pair_merge}')


def l2_norm(x, axis=None):
    """
    takes an input tensor and returns the l2 norm along specified axis
    """

    square_sum = K.sum(K.square(x), axis=axis, keepdims=True)
    norm = K.sqrt(K.maximum(square_sum, K.epsilon()))

    return norm


def pairwise_cosine_difference(t1, t2):
    t1_norm = t1 / l2_norm(t1, axis=-1)
    t2_norm = t2 / l2_norm(t2, axis=-1)
    dot = K.clip(K.batch_dot(t1_norm, t2_norm), -1, 1)
    return K.mean(tf.acos(dot))


def contrastive_difference(left: Tensor, right: Tensor, batch_size: int, temperature: Tensor):
    left_normed = left / l2_norm(left, axis=-1)
    right_normed = right / l2_norm(right, axis=-1)
    logits_left = tf.linalg.matmul(left_normed, right_normed, transpose_b=True) * tf.math.exp(temperature)
    logits_right = tf.linalg.matmul(right_normed, left_normed, transpose_b=True) * tf.math.exp(temperature)
    prob_left = tf.keras.activations.softmax(logits_left, axis=-1)
    prob_right = tf.keras.activations.softmax(logits_right, axis=-1)

    # identity matrix (np.eye) matches left row modality with right column modality
    labels = tf.convert_to_tensor(np.eye(batch_size), dtype=tf.float32)
    loss_left = tf.keras.losses.CategoricalCrossentropy(from_logits=False, reduction=tf.keras.losses.Reduction.SUM)(prob_left, labels)
    loss_right = tf.keras.losses.CategoricalCrossentropy(from_logits=False, reduction=tf.keras.losses.Reduction.SUM)(prob_right, labels)
    loss = (loss_left + loss_right)/2
    return loss / batch_size


class CosineLossLayer(Layer):
    """Layer that creates an Cosine loss."""

    def __init__(self, weight, **kwargs):
        super(CosineLossLayer, self).__init__(**kwargs)
        self.weight = weight

    def get_config(self):
        config = super().get_config().copy()
        config.update({'weight': self.weight})
        return config

    def call(self, inputs):
        # We use `add_loss` to create a regularization loss
        # that depends on the inputs.
        self.add_loss(self.weight * pairwise_cosine_difference(inputs[0], inputs[1]))
        return inputs


class L2LossLayer(Layer):
    """Layer that creates an L2 loss."""

    def __init__(self, weight, **kwargs):
        super(L2LossLayer, self).__init__(**kwargs)
        self.weight = weight

    def get_config(self):
        config = super().get_config().copy()
        config.update({'weight': self.weight})
        return config

    def call(self, inputs):
        #self.add_loss(self.weight * tf.reduce_sum(tf.square(inputs[0] - inputs[1])))
        self.add_loss(self.weight * K.mean(l2_norm(inputs[0] - inputs[1])))
        return inputs


class ContrastiveLossLayer(Layer):
    """Layer that creates an Cosine loss."""

    def __init__(self, weight, batch_size, **kwargs):
        super(ContrastiveLossLayer, self).__init__(**kwargs)
        self.weight = weight
        self.batch_size = batch_size
        self.temperature = self.add_weight(
            name='contrastive_temperature',
            shape=(1,), initializer="zeros", trainable=True,
        )

    def get_config(self):
        config = super().get_config().copy()
        config.update({'weight': self.weight, 'batch_size': self.batch_size})
        return config

    def call(self, inputs):
        # We use `add_loss` to create a regularization loss
        # that depends on the inputs.
        contrastive_loss = self.weight * contrastive_difference(inputs[0], inputs[1], self.batch_size, self.temperature)
        self.add_loss(contrastive_loss)
        self.add_metric(contrastive_loss, name="contrastive_loss")
        return inputs


class LinearTransform(tf.keras.layers.Layer):
    """Layer that implements y=m*x+b, where m and b are
    learnable parameters.
    """
    def __init__(
        self,
        gamma_initializer="ones",
        beta_initializer="zeros",
        dtype=None,
        **kwargs
    ):
        super().__init__(dtype=dtype, **kwargs)
        self.gamma_initializer = gamma_initializer
        self.beta_initializer = beta_initializer

    def build(self, input_shape):
        num_channels = int(input_shape[-1])
        self.gamma = self.add_weight(
            "gamma",
            shape=[num_channels],
            initializer=self.gamma_initializer,
            dtype=self.dtype,
        )
        self.beta = self.add_weight(
            "beta",
            shape=[num_channels],
            initializer=self.beta_initializer,
            dtype=self.dtype,
        )

    def call(self, inputs):
        return self.gamma * inputs[0] + self.beta

# class ContrastiveLossLayer(Layer):
#     """Layer that creates a Contrastive between modalities"""
#
#     def __init__(self, weight, batch_size, **kwargs):
#         super(ContrastiveLossLayer, self).__init__(**kwargs)
#         self.weight = weight
#         self.batch_size = batch_size
#         self.temperature = self.add_weight(shape=(1,), initializer="zeros", trainable=True)
#
#     def get_config(self):
#         config = super().get_config().copy()
#         config.update({'weight': self.weight, 'batch_size': self.batch_size})
#         return config
#
#     def call(self, inputs):
#         self.add_loss(self.weight * contrastive_difference(inputs[0], inputs[1], self.batch_size, self.temperature))
#         return inputs


class VariationalDiagNormal(Layer):
    def __init__(
            self,
            latent_size: int,
            kl_divergence_weight: float = 1.,
            **kwargs
    ):
        self.latent_size = latent_size
        self.kl_divergence_weight = kl_divergence_weight
        super(VariationalDiagNormal, self).__init__(**kwargs)
        self.prior = tfd.MultivariateNormalDiag(loc=tf.zeros([latent_size]), scale_identity_multiplier=1.0)

    def call(self, mu: Tensor, log_sigma: Tensor, **kwargs):
        """mu and sigma must be shape (None, latent_size)"""
        approx_posterior = tfd.MultivariateNormalDiag(loc=mu, scale_diag=tf.math.exp(log_sigma))
        kl = tf.reduce_mean(tfd.kl_divergence(approx_posterior, self.prior))
        self.add_loss(kl * self.kl_divergence_weight)
        self.add_metric(kl, name='KL_divergence')
        return approx_posterior.sample()

    def get_config(self):
        return {'latent_size': self.latent_size, 'kl_divergence_weight': self.kl_divergence_weight}
