import datetime
import logging
import os
import math

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

import keras
from keras import layers
from keras.models import load_model
from keras.saving import register_keras_serializable

from ml4h.defines import IMAGE_EXT
from ml4h.TensorMap import TensorMap, Interpretation
from ml4h.metrics import KernelInceptionDistance, MultiScaleSSIM

Tensor = tf.Tensor

# sampling
min_signal_rate = 0.02
max_signal_rate = 0.98

# architecture
embedding_dims = 256
embedding_max_frequency = 1000.0

# optimization
ema = 0.999

# plotting - increased for higher quality
plot_diffusion_steps = 100

# attention settings
DEFAULT_ATTENTION_RESOLUTIONS = [32, 16, 8]  # apply attention at these spatial resolutions

@register_keras_serializable()
def sinusoidal_embedding(x, dims=1):
    embedding_min_frequency = 1.0
    frequencies = tf.exp(
        tf.linspace(
            tf.math.log(embedding_min_frequency),
            tf.math.log(embedding_max_frequency),
            embedding_dims // 2,
        ),
    )
    angular_speeds = 2.0 * math.pi * frequencies
    if dims == 1:
        embeddings = tf.concat([tf.sin(angular_speeds * x)], axis=2)
    elif dims == 2:
        embeddings = tf.concat([tf.sin(angular_speeds * x), tf.cos(angular_speeds * x)], axis=3)
    elif dims == 3:
        embeddings = tf.concat(
            [tf.sin(angular_speeds * x), tf.cos(angular_speeds * x), -tf.sin(angular_speeds * x)],
            axis=4,
        )
    else:
        raise ValueError(f'No support for 4d or more.')
    return embeddings

def get_norm_groups(channels, max_groups=32):
    """Dynamically compute number of groups for GroupNorm based on channel count."""
    for g in [max_groups, 16, 8, 4, 2, 1]:
        if channels % g == 0 and channels >= g:
            return g
    return 1


@register_keras_serializable()
class SelfAttention2D(layers.Layer):
    """Multi-head self-attention for 2D feature maps.

    Critical for capturing long-range dependencies in larger images.
    Applied at lower resolutions (e.g., 32x32, 16x16, 8x8) for efficiency.
    """
    def __init__(self, channels, num_heads=8, **kwargs):
        super().__init__(**kwargs)
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = max(channels // num_heads, 1)

    def build(self, input_shape):
        self.norm = tf.keras.layers.GroupNormalization(
            groups=get_norm_groups(self.channels), axis=-1
        )
        self.qkv = layers.Dense(self.channels * 3, use_bias=False)
        self.proj = layers.Dense(self.channels)
        super().build(input_shape)

    def call(self, x):
        batch_size = tf.shape(x)[0]
        h, w = x.shape[1], x.shape[2]

        # Normalize and reshape for attention
        x_norm = self.norm(x)
        x_flat = tf.reshape(x_norm, [batch_size, h * w, self.channels])

        # Compute Q, K, V
        qkv = self.qkv(x_flat)
        qkv = tf.reshape(qkv, [batch_size, h * w, 3, self.num_heads, self.head_dim])
        qkv = tf.transpose(qkv, [2, 0, 3, 1, 4])  # [3, B, heads, seq, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Scaled dot-product attention
        scale = tf.math.rsqrt(tf.cast(self.head_dim, tf.float32))
        attn = tf.matmul(q, k, transpose_b=True) * scale
        attn = tf.nn.softmax(attn, axis=-1)

        # Apply attention to values
        out = tf.matmul(attn, v)
        out = tf.transpose(out, [0, 2, 1, 3])  # [B, seq, heads, head_dim]
        out = tf.reshape(out, [batch_size, h * w, self.channels])
        out = self.proj(out)
        out = tf.reshape(out, [batch_size, h, w, self.channels])

        return x + out  # Residual connection

    def get_config(self):
        config = super().get_config()
        config.update({
            'channels': self.channels,
            'num_heads': self.num_heads,
        })
        return config


@register_keras_serializable()
class SelfAttention1D(layers.Layer):
    """Multi-head self-attention for 1D sequences (e.g., ECGs)."""
    def __init__(self, channels, num_heads=8, **kwargs):
        super().__init__(**kwargs)
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = max(channels // num_heads, 1)

    def build(self, input_shape):
        self.norm = tf.keras.layers.GroupNormalization(
            groups=get_norm_groups(self.channels), axis=-1
        )
        self.qkv = layers.Dense(self.channels * 3, use_bias=False)
        self.proj = layers.Dense(self.channels)
        super().build(input_shape)

    def call(self, x):
        batch_size = tf.shape(x)[0]
        seq_len = x.shape[1]

        # Normalize
        x_norm = self.norm(x)

        # Compute Q, K, V
        qkv = self.qkv(x_norm)
        qkv = tf.reshape(qkv, [batch_size, seq_len, 3, self.num_heads, self.head_dim])
        qkv = tf.transpose(qkv, [2, 0, 3, 1, 4])
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Scaled dot-product attention
        scale = tf.math.rsqrt(tf.cast(self.head_dim, tf.float32))
        attn = tf.matmul(q, k, transpose_b=True) * scale
        attn = tf.nn.softmax(attn, axis=-1)

        # Apply attention
        out = tf.matmul(attn, v)
        out = tf.transpose(out, [0, 2, 1, 3])
        out = tf.reshape(out, [batch_size, seq_len, self.channels])
        out = self.proj(out)

        return x + out

    def get_config(self):
        config = super().get_config()
        config.update({
            'channels': self.channels,
            'num_heads': self.num_heads,
        })
        return config


@register_keras_serializable()
def residual_block(width, conv, kernel_size, groups=32):
    def apply(x):
        # shortcut
        input_channels = x.shape[-1]
        if input_channels != width:
            residual = conv(width, kernel_size=1)(x)
        else:
            residual = x

        # Dynamic groups based on channel count
        num_groups = get_norm_groups(width, groups)

        # first GN → SiLU → Conv
        x = tf.keras.layers.GroupNormalization(groups=num_groups, axis=-1)(x)
        x = layers.Activation('silu')(x)
        x = conv(width, kernel_size=kernel_size, padding="same")(x)

        # second GN → SiLU → Conv
        x = tf.keras.layers.GroupNormalization(groups=num_groups, axis=-1)(x)
        x = layers.Activation('silu')(x)
        x = conv(width, kernel_size=kernel_size, padding="same")(x)

        # merge
        x = layers.Add()([x, residual])
        return x
    return apply

@register_keras_serializable()
def down_block(width, block_depth, conv, pool, kernel_size, groups=32, use_attention=False, attention_heads=8, ndim=2):
    def apply(x):
        x, skips = x
        for _ in range(block_depth):
            x = residual_block(width, conv, kernel_size, groups=groups)(x)
            skips.append(x)
        # Apply self-attention after residual blocks if enabled
        if use_attention:
            if ndim == 2:
                x = SelfAttention2D(width, num_heads=attention_heads)(x)
            elif ndim == 1:
                x = SelfAttention1D(width, num_heads=attention_heads)(x)
        x = pool(pool_size=2)(x)
        return x
    return apply

@register_keras_serializable()
def up_block(width, block_depth, conv, upsample, kernel_size, groups=32, use_attention=False, attention_heads=8, ndim=2):
    def apply(x):
        x, skips = x
        x = upsample(size=2)(x)
        for _ in range(block_depth):
            x = layers.Concatenate()([x, skips.pop()])
            x = residual_block(width, conv, kernel_size, groups=groups)(x)
        # Apply self-attention after residual blocks if enabled
        if use_attention:
            if ndim == 2:
                x = SelfAttention2D(width, num_heads=attention_heads)(x)
            elif ndim == 1:
                x = SelfAttention1D(width, num_heads=attention_heads)(x)
        return x
    return apply

@register_keras_serializable()
def get_network(input_shape, widths, block_depth, kernel_size, attention_resolutions=None, attention_heads=8):
    """Build a U-Net with optional self-attention at specified resolutions.

    Args:
        input_shape: Shape of input images (H, W, C) or (L, C) for 1D
        widths: List of channel widths for each U-Net level
        block_depth: Number of residual blocks per level
        kernel_size: Convolution kernel size
        attention_resolutions: List of spatial resolutions where attention is applied.
                              For 256x256 images, typical values are [32, 16, 8].
                              None or empty list disables attention.
        attention_heads: Number of attention heads
    """
    if attention_resolutions is None:
        attention_resolutions = DEFAULT_ATTENTION_RESOLUTIONS

    noisy_images = keras.Input(shape=input_shape)
    conv, upsample, pool, _ = layers_from_shape_control(input_shape)
    noise_variances = keras.Input(shape=[1] * len(input_shape))

    # Determine dimensionality (1D for ECG, 2D for images)
    ndim = len(input_shape) - 1  # subtract channel dim

    e = layers.Lambda(sinusoidal_embedding)(noise_variances)
    if len(input_shape) == 2:
        e = upsample(size=input_shape[-2])(e)
    else:
        e = upsample(size=input_shape[:-1], interpolation="nearest")(e)

    print(f'e shape: {e.shape} len {len(input_shape)}')
    x = conv(widths[0], kernel_size=1)(noisy_images)
    x = layers.Concatenate()([x, e])

    # Track spatial resolution for attention decisions
    current_resolution = input_shape[0]  # Start with input resolution

    skips = []
    for i, width in enumerate(widths[:-1]):
        # Check if we should apply attention at this resolution
        use_attention = (current_resolution in attention_resolutions) and ndim <= 2
        x = down_block(
            width, block_depth, conv, pool, kernel_size,
            use_attention=use_attention, attention_heads=attention_heads, ndim=ndim
        )([x, skips])
        current_resolution = current_resolution // 2

    # Bottleneck with attention (always apply attention at bottleneck for larger images)
    use_bottleneck_attention = (current_resolution <= 32) and ndim <= 2
    for _ in range(block_depth):
        x = residual_block(widths[-1], conv, kernel_size)(x)
    if use_bottleneck_attention:
        if ndim == 2:
            x = SelfAttention2D(widths[-1], num_heads=attention_heads)(x)
        elif ndim == 1:
            x = SelfAttention1D(widths[-1], num_heads=attention_heads)(x)

    # Up path - mirror the attention pattern
    for i, width in enumerate(reversed(widths[:-1])):
        current_resolution = current_resolution * 2
        use_attention = (current_resolution in attention_resolutions) and ndim <= 2
        x = up_block(
            width, block_depth, conv, upsample, kernel_size,
            use_attention=use_attention, attention_heads=attention_heads, ndim=ndim
        )([x, skips])

    x = conv(input_shape[-1], kernel_size=1, kernel_initializer="zeros")(x)

    return keras.Model([noisy_images, noise_variances], x, name="residual_unet")


def layers_from_shape_control(input_shape):
    if len(input_shape) == 2:
        return layers.Conv1D, layers.UpSampling1D, layers.AveragePooling1D, tuple(
            [slice(None), np.newaxis, slice(None)],
        )
    elif len(input_shape) == 3:
        return layers.Conv2D, layers.UpSampling2D, layers.AveragePooling2D, tuple(
            [slice(None), np.newaxis, np.newaxis, slice(None)],
        )
    elif len(input_shape) == 4:
        return layers.Conv3D, layers.UpSampling3D, layers.AveragePooling3D, tuple(
            [slice(None), np.newaxis, np.newaxis, np.newaxis, slice(None)],
        )


def condition_layer_film(input_tensor, control_vector, filters):
    # Transform control into gamma and beta
    gamma = layers.Dense(filters, activation="linear")(control_vector)
    beta = layers.Dense(filters, activation="linear")(control_vector)

    # Reshape gamma and beta to match the spatial dimensions
    # if 4 == len(input_tensor.shape):
    #     gamma = tf.reshape(gamma, (-1, 1, 1, filters))
    #     beta = tf.reshape(beta, (-1, 1, 1, filters))
    # elif 3 == len(input_tensor.shape):
    #     gamma = tf.reshape(gamma, (-1, 1, filters))
    #     beta = tf.reshape(beta, (-1, 1, filters))
    # Apply FiLM (Feature-wise Linear Modulation)
    return input_tensor * gamma + beta


def residual_block_control(
    width,
    conv,
    kernel_size,
    attention_heads,
    condition_strategy,
    groups: int = 32,
):
    def apply(inputs):
        x, control = inputs
        # ─── shortcut ─────────────────────────────────────────────
        in_channels = x.shape[-1]
        if in_channels != width:
            residual = conv(width, kernel_size=1)(x)
        else:
            residual = x

        # ─── conditioning ─────────────────────────────────────────
        if condition_strategy == 'cross_attention':
            x = layers.MultiHeadAttention(
                num_heads=attention_heads, key_dim=width
            )(x, control)
        elif condition_strategy == 'concat':
            x = layers.Concatenate()([x, control])
        elif condition_strategy == 'film':
            x = condition_layer_film(x, control, width)

        # ─── GN → SiLU → Conv → GN → SiLU → Conv → Add ───────────
        x = tf.keras.layers.GroupNormalization(groups=groups, axis=-1)(x)
        x = layers.Activation('silu')(x)
        x = conv(width, kernel_size=kernel_size, padding='same')(x)
        x = tf.keras.layers.GroupNormalization(groups=groups, axis=-1)(x)
        x = layers.Activation('silu')(x)
        x = conv(width, kernel_size=kernel_size, padding='same')(x)

        x = layers.Add()([x, residual])
        return x

    return apply


def down_block_control(
    width,
    block_depth,
    conv,
    pool,
    kernel_size,
    attention_heads,
    condition_strategy,
    groups: int = 32,
):
    def apply(inputs):
        x, skips, control = inputs
        for _ in range(block_depth):
            x = residual_block_control(
                width,
                conv,
                kernel_size,
                attention_heads,
                condition_strategy,
                groups=groups,
            )([x, control])
            skips.append(x)
        x = pool(pool_size=2)(x)
        return x

    return apply


def up_block_control(
    width,
    block_depth,
    conv,
    upsample,
    kernel_size,
    attention_heads,
    condition_strategy,
    groups: int = 32,
):
    def apply(inputs):
        x, skips, control = inputs
        x = upsample(size=2)(x)
        for _ in range(block_depth):
            x = layers.Concatenate()([x, skips.pop()])
            x = residual_block_control(
                width,
                conv,
                kernel_size,
                attention_heads,
                condition_strategy,
                groups=groups,
            )([x, control])
        return x

    return apply


def get_control_network(
    input_shape,
    widths,
    block_depth,
    kernel_size,
    control_size,
    attention_window,
    attention_heads,
    attention_modulo,
    condition_strategy,
    groups: int = 32,
):
    noisy_images = keras.Input(shape=input_shape)
    noise_variances = keras.Input(shape=[1] * len(input_shape))
    conv, upsample, pool, control_idxs = layers_from_shape_control(input_shape)
    control = keras.Input(shape=(control_size,))

    x = conv(widths[0], kernel_size=1)(noisy_images)
    e = layers.Lambda(sinusoidal_embedding)(noise_variances)

    # upsample embeddings & control
    if len(input_shape) == 2:
        e = upsample(size=input_shape[-2])(e)
        c = upsample(size=input_shape[-2])(control[control_idxs])
    else:
        e = upsample(size=input_shape[:-1], interpolation="nearest")(e)
        c = upsample(size=input_shape[:-1])(control[control_idxs])

    x = layers.Concatenate()([x, e])

    skips = []
    # ─── down blocks ────────────────────────────────────────────
    for i, width in enumerate(widths[:-1]):
        use_control = (attention_modulo > 1 and (i+1) % attention_modulo == 0)
        if use_control:
            # recompute c2 at this resolution...
            c2 = upsample(size=x.shape[1:-1] if len(input_shape)>2 else x.shape[-2])(control[control_idxs])
            x = down_block_control(
                width,
                block_depth,
                conv,
                pool,
                kernel_size,
                attention_heads,
                condition_strategy,
                groups=groups,
            )([x, skips, c2])
        else:
            x = down_block(width, block_depth, conv, pool, kernel_size)([x, skips])

    # ─── bottleneck ────────────────────────────────────────────
    c2 = upsample(size=x.shape[1:-1] if len(input_shape)>2 else x.shape[-2])(control[control_idxs])
    for _ in range(block_depth):
        x = residual_block_control(
            widths[-1],
            conv,
            kernel_size,
            attention_heads,
            condition_strategy,
            groups=groups,
        )([x, c2])

    # ─── up blocks ──────────────────────────────────────────────
    for i, width in enumerate(reversed(widths[:-1])):
        use_control = (attention_modulo > 1 and (len(widths)-1 - i) % attention_modulo == 0)
        if use_control:
            up_size = (x.shape[1]*2, x.shape[2]*2) if len(input_shape)==3 else x.shape[-2]*2
            c2 = upsample(size=up_size)(control[control_idxs])
            x = up_block_control(
                width,
                block_depth,
                conv,
                upsample,
                kernel_size,
                attention_heads,
                condition_strategy,
                groups=groups,
            )([x, skips, c2])
        else:
            x = up_block(width, block_depth, conv, upsample, kernel_size)([x, skips])

    # ─── final convs ────────────────────────────────────────────
    x = conv(input_shape[-1], kernel_size=1, activation="linear", kernel_initializer="zeros")(x)

    return keras.Model([noisy_images, noise_variances, control], x, name="control_unet")


def get_control_embed_model(output_maps, control_size):
    control_ins = []
    logging.info(f'Build control embedder on :{output_maps}')
    for cm in output_maps:
        control_ins.append(keras.Input(shape=cm.shape, name=cm.output_name()))
    c = layers.Concatenate()(control_ins)
    c = layers.Dense(control_size, activation='linear')(c)
    return keras.Model(control_ins, c, name='control_embed')

@register_keras_serializable()
class DiffusionModel(keras.Model):
    def __init__(self, tensor_map, batch_size, widths, block_depth, kernel_size, diffusion_loss, sigmoid_beta, inspect_model,
                 attention_resolutions=None, attention_heads=8, prediction_type='v', sampler='ddim',
                 name=None,
                 **kwargs):
        """Diffusion model for image and waveform generation.

        Args:
            tensor_map: TensorMap defining input shape and properties
            batch_size: Training batch size
            widths: List of channel widths for U-Net levels
            block_depth: Number of residual blocks per level
            kernel_size: Convolution kernel size
            diffusion_loss: Loss type ('mse' or 'sigmoid')
            sigmoid_beta: Beta parameter for sigmoid loss weighting
            inspect_model: Whether to compute expensive metrics during validation
            attention_resolutions: Spatial resolutions for self-attention (e.g., [32, 16, 8]).
                                   For 256x256 images, attention at these resolutions is critical.
                                   Set to [] to disable attention (faster but lower quality).
            attention_heads: Number of attention heads (default 8)
            prediction_type: 'eps' for noise prediction, 'v' for v-prediction (better for high resolution)
            sampler: 'ddpm' for original sampler, 'ddim' for deterministic higher-quality sampling
        """
        super().__init__()

        self.tensor_map     = tensor_map
        self.batch_size     = batch_size
        self.widths         = widths
        self.block_depth    = block_depth
        self.kernel_size    = kernel_size
        self.diffusion_loss = diffusion_loss
        self.sigmoid_beta   = sigmoid_beta
        self.inspect_model  = False  # inspect_model
        self.attention_resolutions = attention_resolutions if attention_resolutions is not None else DEFAULT_ATTENTION_RESOLUTIONS
        self.attention_heads = attention_heads
        self.prediction_type = prediction_type
        self.sampler = sampler

        self.normalizer = layers.Normalization()
        self.network = get_network(
            self.tensor_map.shape, widths, block_depth, kernel_size,
            attention_resolutions=self.attention_resolutions,
            attention_heads=self.attention_heads
        )
        self.ema_network = keras.models.clone_model(self.network)
        self.use_sigmoid_loss = diffusion_loss == 'sigmoid'

    def can_apply(self):
        return self.tensor_map.axes() > 1

    def get_config(self):
        config = super().get_config()
        # pop out any Keras-internal stuff you don’t want to pass back
        config.pop("layers", None)
        config.pop("input_layers", None)
        config.pop("output_layers", None)
        # now re-inject exactly the args your __init__ needs:
        logging.info(f'Saving tensormap as: {str(self.tensor_map)}')
        config.update({
            "tensor_map":     str(self.tensor_map),        # or .to_config() if needed
            "batch_size":     self.batch_size,
            "widths":         self.widths,
            "block_depth":    self.block_depth,
            "kernel_size":    self.kernel_size,
            "diffusion_loss": self.diffusion_loss,
            "sigmoid_beta":   self.sigmoid_beta,
            "inspect_model":  self.inspect_model,
            "attention_resolutions": self.attention_resolutions,
            "attention_heads": self.attention_heads,
            "prediction_type": self.prediction_type,
            "sampler": self.sampler,
            # name/trainable/dtype are already in super().get_config()
        })
        return config

    @classmethod
    def from_config(cls, config):
        tm_string = config.pop("tensor_map")
        config['tensor_map'] = TensorMap('lax_4ch_random_slice_3d', Interpretation.CONTINUOUS, shape=(160, 224, 1))
        return cls(**config)

    def compile(self, **kwargs):
        super().compile(**kwargs)

        self.noise_loss_tracker = keras.metrics.Mean(name="n_loss")
        self.image_loss_tracker = keras.metrics.Mean(name="i_loss")
        self.mse_metric = tf.keras.metrics.MeanSquaredError(name="mse")
        self.mae_metric = tf.keras.metrics.MeanAbsoluteError(name="mae")
        if self.tensor_map.axes() == 3 and self.inspect_model:
            self.kid = KernelInceptionDistance(name = "kid", input_shape = self.tensor_map.shape, kernel_image_size=299)
            self.ms_ssim = MultiScaleSSIM()

    @property
    def metrics(self):
        m = [self.noise_loss_tracker, self.image_loss_tracker, self.mse_metric, self.mae_metric]
        if self.tensor_map.axes() == 3 and self.inspect_model:
            m.append(self.kid)
            m.append(self.ms_ssim)
        return m

    def denormalize(self, images):
        with tf.init_scope():
            mean = self.normalizer.mean
            var = self.normalizer.variance
        std = tf.sqrt(var)
        return images * std + mean
        # images = self.normalizer.mean + images * self.normalizer.variance ** 0.5
        # # print(f'images max min {images}')
        # return images  # tf.clip_by_value(images, 0.0, 1.0)

    def diffusion_schedule(self, diffusion_times):
        # diffusion times -> angles
        start_angle = tf.acos(max_signal_rate)
        end_angle = tf.acos(min_signal_rate)

        diffusion_angles = start_angle + diffusion_times * (end_angle - start_angle)

        # angles -> signal and noise rates
        signal_rates = tf.cos(diffusion_angles)
        noise_rates = tf.sin(diffusion_angles)
        # note that their squared sum is always: sin^2(x) + cos^2(x) = 1

        return noise_rates, signal_rates

    def denoise(self, noisy_images, noise_rates, signal_rates, training):
        """Denoise images using the network prediction.

        Supports both epsilon (noise) prediction and v-prediction parameterizations.
        V-prediction often produces better results for high-resolution images.
        """
        # the exponential moving average weights are used at evaluation
        if training:
            network = self.network
        else:
            network = self.ema_network

        # Network predicts either noise (eps) or velocity (v)
        pred = network([noisy_images, noise_rates ** 2], training=training)

        if self.prediction_type == 'v':
            # V-prediction: network predicts v = signal_rate * noise - noise_rate * image
            # Recover noise and image from v-prediction
            pred_images = signal_rates * noisy_images - noise_rates * pred
            pred_noises = noise_rates * noisy_images + signal_rates * pred
        else:
            # Epsilon prediction (default): network predicts noise directly
            pred_noises = pred
            pred_images = (noisy_images - noise_rates * pred_noises) / signal_rates

        return pred_noises, pred_images

    def reverse_diffusion_ddpm(self, initial_noise, diffusion_steps):
        """Original DDPM sampling (stochastic)."""
        num_images = initial_noise.shape[0]
        step_size = 1.0 / diffusion_steps

        next_noisy_images = initial_noise
        for step in range(diffusion_steps):
            noisy_images = next_noisy_images

            diffusion_times = tf.ones([num_images] + [1] * self.tensor_map.axes()) - step * step_size
            noise_rates, signal_rates = self.diffusion_schedule(diffusion_times)
            pred_noises, pred_images = self.denoise(
                noisy_images, noise_rates, signal_rates, training=False,
            )

            next_diffusion_times = diffusion_times - step_size
            next_noise_rates, next_signal_rates = self.diffusion_schedule(
                next_diffusion_times,
            )
            next_noisy_images = (
                    next_signal_rates * pred_images + next_noise_rates * pred_noises
            )

        return pred_images

    def reverse_diffusion_ddim(self, initial_noise, diffusion_steps, eta=0.0):
        """DDIM sampling (deterministic when eta=0, better quality for fewer steps).

        DDIM produces higher quality samples especially with fewer diffusion steps.
        Set eta=0 for fully deterministic sampling, eta=1 for DDPM-like stochasticity.

        Reference: "Denoising Diffusion Implicit Models" (Song et al., 2020)
        """
        num_images = initial_noise.shape[0]
        step_size = 1.0 / diffusion_steps

        next_noisy_images = initial_noise
        pred_images = None

        for step in range(diffusion_steps):
            noisy_images = next_noisy_images

            # Current timestep
            diffusion_times = tf.ones([num_images] + [1] * self.tensor_map.axes()) - step * step_size
            noise_rates, signal_rates = self.diffusion_schedule(diffusion_times)

            # Get model prediction
            pred_noises, pred_images = self.denoise(
                noisy_images, noise_rates, signal_rates, training=False,
            )

            # Next timestep
            next_diffusion_times = diffusion_times - step_size
            next_noise_rates, next_signal_rates = self.diffusion_schedule(
                next_diffusion_times,
            )

            # DDIM update rule
            # x_{t-1} = sqrt(alpha_{t-1}) * x0_pred + sqrt(1 - alpha_{t-1} - sigma^2) * eps_pred + sigma * noise
            # where sigma = eta * sqrt((1 - alpha_{t-1}) / (1 - alpha_t)) * sqrt(1 - alpha_t / alpha_{t-1})

            if eta > 0 and step < diffusion_steps - 1:
                # Compute sigma for stochastic sampling
                sigma = eta * tf.sqrt(
                    (next_noise_rates ** 2 / (noise_rates ** 2 + 1e-8)) *
                    (1.0 - (signal_rates ** 2) / (next_signal_rates ** 2 + 1e-8))
                )
                # Direction pointing to x_t
                dir_xt = tf.sqrt(tf.maximum(next_noise_rates ** 2 - sigma ** 2, 0.0)) * pred_noises
                # Add noise
                noise = tf.random.normal(shape=tf.shape(noisy_images))
                next_noisy_images = next_signal_rates * pred_images + dir_xt + sigma * noise
            else:
                # Deterministic DDIM (eta=0)
                next_noisy_images = next_signal_rates * pred_images + next_noise_rates * pred_noises

        return pred_images

    def reverse_diffusion(self, initial_noise, diffusion_steps):
        """Main reverse diffusion method - dispatches to selected sampler."""
        if self.sampler == 'ddim':
            return self.reverse_diffusion_ddim(initial_noise, diffusion_steps, eta=0.0)
        else:
            return self.reverse_diffusion_ddpm(initial_noise, diffusion_steps)

    def generate(self, num_images, diffusion_steps, reseed=None, eta=0.0):
        """Generate synthetic images/waveforms from noise.

        Args:
            num_images: Number of samples to generate
            diffusion_steps: Number of denoising steps. For high quality:
                            - 256x256 color images: use 100-200 steps
                            - Smaller grayscale: 50-100 steps may suffice
            reseed: Optional random seed for reproducibility
            eta: DDIM stochasticity parameter (only used when sampler='ddim'):
                 - eta=0: Fully deterministic (recommended for quality)
                 - eta=1: Same stochasticity as DDPM
                 - 0 < eta < 1: Interpolation between the two

        Returns:
            Generated and denormalized images/waveforms
        """
        if reseed is not None:
            tf.random.set_seed(reseed)
        initial_noise = tf.random.normal(shape=(num_images,) + self.tensor_map.shape)

        if self.sampler == 'ddim':
            generated_images = self.reverse_diffusion_ddim(initial_noise, diffusion_steps, eta=eta)
        else:
            generated_images = self.reverse_diffusion_ddpm(initial_noise, diffusion_steps)

        generated_images = self.denormalize(generated_images)
        return generated_images

    def train_step(self, images_original):
        # normalize images to have standard deviation of 1, like the noises
        images = images_original[0][self.tensor_map.input_name()]
        images = self.normalizer(images, training=True)
        noises = tf.random.normal(shape=(self.batch_size,) + self.tensor_map.shape)

        # sample uniform random diffusion times
        diffusion_times = tf.random.uniform(
            shape=[self.batch_size] + [1] * self.tensor_map.axes(), minval=0.0, maxval=1.0,
        )
        noise_rates, signal_rates = self.diffusion_schedule(diffusion_times)
        # mix the images with noises accordingly
        print(f'noises.shape {noises.shape} images.shape {images.shape}')
        noisy_images = signal_rates * images + noise_rates * noises

        with tf.GradientTape() as tape:
            # Get network prediction (either noise or velocity depending on prediction_type)
            if self.prediction_type == 'eps':
                network = self.network
            else:
                network = self.network

            pred = network([noisy_images, noise_rates ** 2], training=True)

            if self.prediction_type == 'v':
                # V-prediction: target is v = signal_rate * noise - noise_rate * image
                # This parameterization has better gradient flow at high noise levels
                target_v = signal_rates * noises - noise_rates * images
                prediction_loss = self.loss(target_v, pred)
                # Recover pred_noises and pred_images for metrics
                pred_images = signal_rates * noisy_images - noise_rates * pred
                pred_noises = noise_rates * noisy_images + signal_rates * pred
            else:
                # Epsilon prediction: target is the noise
                prediction_loss = self.loss(noises, pred)
                pred_noises = pred
                pred_images = (noisy_images - noise_rates * pred_noises) / signal_rates

            noise_loss = prediction_loss
            image_loss = self.loss(images, pred_images)  # only used as metric
            if self.use_sigmoid_loss:
                signal_rates_squared = tf.square(signal_rates)
                noise_rates_squared = tf.square(noise_rates)

                # Compute log-SNR (lambda_t)
                lambda_t = tf.math.log(signal_rates_squared / noise_rates_squared)
                weight = tf.math.sigmoid(self.sigmoid_beta - lambda_t)
                noise_loss = weight * noise_loss

        gradients = tape.gradient(noise_loss, self.network.trainable_weights)
        self.optimizer.apply_gradients(zip(gradients, self.network.trainable_weights))

        self.noise_loss_tracker.update_state(noise_loss)
        self.image_loss_tracker.update_state(image_loss)
        self.mse_metric.update_state(noises, pred_noises)
        self.mae_metric.update_state(noises, pred_noises)

        # track the exponential moving averages of weights
        for weight, ema_weight in zip(self.network.weights, self.ema_network.weights):
            ema_weight.assign(ema * ema_weight + (1 - ema) * weight)

        # KID is not measured during the training phase for computational efficiency
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, images_original):
        # normalize images to have standard deviation of 1, like the noises
        images = images_original[0][self.tensor_map.input_name()]
        images = self.normalizer(images, training=False)
        noises = tf.random.normal(shape=(self.batch_size,) + self.tensor_map.shape)

        # sample uniform random diffusion times
        diffusion_times = tf.random.uniform(
            shape=[self.batch_size] + [1] * self.tensor_map.axes(), minval=0.0, maxval=1.0,
        )
        noise_rates, signal_rates = self.diffusion_schedule(diffusion_times)
        # mix the images with noises accordingly
        noisy_images = signal_rates * images + noise_rates * noises

        # Get network prediction
        pred = self.ema_network([noisy_images, noise_rates ** 2], training=False)

        if self.prediction_type == 'v':
            # V-prediction: compute loss against velocity target
            target_v = signal_rates * noises - noise_rates * images
            noise_loss = self.loss(target_v, pred)
            # Recover pred_noises and pred_images for metrics
            pred_images = signal_rates * noisy_images - noise_rates * pred
            pred_noises = noise_rates * noisy_images + signal_rates * pred
        else:
            # Epsilon prediction
            noise_loss = self.loss(noises, pred)
            pred_noises = pred
            pred_images = (noisy_images - noise_rates * pred_noises) / signal_rates

        image_loss = self.loss(images, pred_images)
        if self.use_sigmoid_loss:
            signal_rates_squared = tf.square(signal_rates)
            noise_rates_squared = tf.square(noise_rates)

            # Compute log-SNR (lambda_t)
            lambda_t = tf.math.log(signal_rates_squared / noise_rates_squared)
            weight = tf.math.sigmoid(self.sigmoid_beta - lambda_t)
            noise_loss = weight * noise_loss

        self.image_loss_tracker.update_state(image_loss)
        self.noise_loss_tracker.update_state(noise_loss)
        self.mse_metric.update_state(noises, pred_noises)
        self.mae_metric.update_state(noises, pred_noises)

        # measure KID between real and generated images
        # this is computationally demanding, kid_diffusion_steps has to be small
        if self.tensor_map.axes() == 3 and self.inspect_model:
            images = self.denormalize(images)
            generated_images = self.generate(
                num_images=self.batch_size, diffusion_steps=50
            )
            self.kid.update_state(images, generated_images)
            self.ms_ssim.update_state(images, generated_images, 255)

        return {m.name: m.result() for m in self.metrics}

    def call(self, inputs, training=False):
        """
        A minimal forward pass so that:
          1. Keras knows how to build() the model
          2. You can use model((noisy_images, noise_rates)) for inference
        """
        noisy_images, noise_rates = inputs
        # re-compute signal_rates
        signal_rates = tf.sqrt(1.0 - tf.square(noise_rates))
        # this returns (pred_noises, pred_images)
        return self.denoise(noisy_images, noise_rates, signal_rates, training=training)

    def plot_images(self, epoch=None, logs=None, num_rows=1, num_cols=4, reseed=None, prefix='./figures/'):
        # plot random generated images for visual evaluation of generation quality
        generated_images = self.generate(
            num_images=num_rows * num_cols,
            diffusion_steps=plot_diffusion_steps,
            reseed=reseed,
        )

        plt.figure(figsize=(num_cols * 2.0, num_rows * 2.0), dpi=300)
        for row in range(num_rows):
            for col in range(num_cols):
                index = row * num_cols + col
                plt.subplot(num_rows, num_cols, index + 1)
                if len(generated_images[index].shape) == 3 and generated_images[index].shape[-1] in [3,4]:
                    img = generated_images[index].numpy()
                    img = (img - img.min()) / (1e-6 + img.max() - img.min())
                    plt.imshow(img)
                elif len(generated_images[index].shape) == 3 and generated_images[index].shape[-1] > 1:
                    plt.imshow(generated_images[index, ..., 0], cmap='gray')  # just plot first frame
                else:
                    plt.imshow(generated_images[index], cmap='gray')
                plt.axis("off")
        plt.tight_layout()
        now_string = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')
        figure_path = os.path.join(prefix, f'diffusion_image_generations_{now_string}{IMAGE_EXT}')
        if not os.path.exists(os.path.dirname(figure_path)):
            os.makedirs(os.path.dirname(figure_path))
        plt.savefig(figure_path, bbox_inches="tight")
        plt.close()

    def plot_ecgs(self, epoch=None, logs=None, num_rows=1, num_cols=4, reseed=None, prefix='./figures/'):
        # plot random generated images for visual evaluation of generation quality
        generated_images = self.generate(
            num_images=max(self.batch_size, num_rows * num_cols),
            diffusion_steps=plot_diffusion_steps,
            reseed=reseed,
        )

        plt.figure(figsize=(num_cols * 2.0, num_rows * 2.0), dpi=300)
        for row in range(num_rows):
            for col in range(num_cols):
                index = row * num_cols + col
                plt.subplot(num_rows, num_cols, index + 1)
                for lead in range(generated_images.shape[-1]):
                    plt.plot(generated_images[index, :, lead], label=lead)
                plt.axis("off")

        plt.tight_layout()
        now_string = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')
        figure_path = os.path.join(prefix, f'diffusion_ecg_generations_{now_string}{IMAGE_EXT}')
        if not os.path.exists(os.path.dirname(figure_path)):
            os.makedirs(os.path.dirname(figure_path))
        plt.savefig(figure_path, bbox_inches="tight")
        plt.close()

    def plot_reconstructions(
        self, images_original, diffusion_amount=0, epoch=None, logs=None, num_rows=2, num_cols=2, prefix='./figures/',
    ):
        images = images_original[0][self.tensor_map.input_name()]
        self.normalizer.adapt(images)
        images = self.normalizer(images, training=False)
        noises = tf.random.normal(shape=(num_rows*num_cols,) + self.tensor_map.shape)

        diffusion_times = diffusion_amount * tf.ones(shape=[num_rows*num_cols] + [1] * self.tensor_map.axes())
        noise_rates, signal_rates = self.diffusion_schedule(diffusion_times)
        # mix the images with noises accordingly
        noisy_images = signal_rates * images + noise_rates * noises

        # use the network to separate noisy images to their components
        pred_noises, generated_images = self.denoise(
            noisy_images, noise_rates, signal_rates, training=False,
        )
        plt.figure(figsize=(num_cols * 2.0, num_rows * 2.0), dpi=300)
        for row in range(num_rows):
            for col in range(num_cols):
                index = row * num_cols + col
                plt.subplot(num_rows, num_cols, index + 1)
                if generated_images.shape[-1] == 1:
                    plt.imshow(generated_images[index], cmap='gray')
                else:
                    img = generated_images[index].numpy()
                    img = (img - img.min()) / (1e-6 + img.max() - img.min())
                    plt.imshow(img)
                plt.axis("off")
        plt.tight_layout()
        now_string = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')
        figure_path = os.path.join(prefix, f'diffusion_reconstructions_{now_string}{IMAGE_EXT}')
        if not os.path.exists(os.path.dirname(figure_path)):
            os.makedirs(os.path.dirname(figure_path))
        plt.savefig(figure_path, bbox_inches="tight")
        plt.close()
        plt.figure(figsize=(num_cols * 2.0, num_rows * 2.0), dpi=300)
        for row in range(num_rows):
            for col in range(num_cols):
                index = row * num_cols + col
                plt.subplot(num_rows, num_cols, index + 1)
                if images.shape[-1] == 1:
                    plt.imshow(images[index], cmap='gray')
                else:
                    img = images[index].numpy()
                    img = (img - img.min()) / (1e-6 + img.max() - img.min())
                    plt.imshow(img)
                plt.axis("off")
        plt.tight_layout()
        now_string = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')
        figure_path = os.path.join(prefix, f'input_images_{now_string}{IMAGE_EXT}')
        if not os.path.exists(os.path.dirname(figure_path)):
            os.makedirs(os.path.dirname(figure_path))
        plt.savefig(figure_path, bbox_inches="tight")
        plt.close()
        return generated_images

    def in_paint(self, images_original, masks, diffusion_steps=64, num_rows=3, num_cols=6):
        images = images_original[0][self.tensor_map.input_name()]
        noises = tf.random.normal(shape=(self.batch_size,) + self.tensor_map.shape)
        # reverse diffusion = sampling
        num_images = images.shape[0]
        step_size = 1.0 / max(0.0001, diffusion_steps)

        # important line:
        # at the first sampling step, the "noisy image" is pure noise
        # but its signal rate is assumed to be nonzero (min_signal_rate)
        next_noisy_images = images * masks + noises * (1 - masks)
        for step in range(diffusion_steps):
            noisy_images = next_noisy_images

            # separate the current noisy image to its components
            diffusion_times = tf.ones([num_images] + [1] * self.tensor_map.axes()) - step * step_size
            noise_rates, signal_rates = self.diffusion_schedule(diffusion_times)
            pred_noises, generated_images = self.denoise(
                noisy_images, noise_rates, signal_rates, training=False,
            )

            # apply the mask
            generated_images = generated_images * (1 - masks) + images * masks

            # remix the predicted components using the next signal and noise rates
            next_diffusion_times = diffusion_times - step_size
            next_noise_rates, next_signal_rates = self.diffusion_schedule(
                next_diffusion_times,
            )
            next_noisy_images = (
                    next_signal_rates * generated_images + next_noise_rates * pred_noises
            )
            # this new noisy image will be used in the next step

        plt.figure(figsize=(num_cols * 2.0, num_rows * 2.0), dpi=300)
        for row in range(num_rows):
            for col in range(num_cols):
                index = row * num_cols + col
                plt.subplot(num_rows, num_cols, index + 1)
                plt.imshow(generated_images[index], cmap='gray')
                plt.axis("off")
        plt.tight_layout()
        plt.show()
        plt.close()

@register_keras_serializable()
class DiffusionController(keras.Model):
    def __init__(
            self, tensor_map, output_maps, batch_size, widths, block_depth, conv_x, control_size,
            attention_start, attention_heads, attention_modulo, diffusion_loss, sigmoid_beta, condition_strategy,
            inspect_model, supervisor=None, supervision_scalar=0.01, encoder_file=None,
    ):
        super().__init__()

        self.input_map = tensor_map
        self.batch_size = batch_size
        self.output_maps = output_maps
        if encoder_file:
            self.autoencoder_control = True
            self.control_embed_model = load_model(encoder_file, compile=False)
            logging.info(f'loaded encoder for DiffAE at: {encoder_file}')
        else:
            self.autoencoder_control = False
            self.control_embed_model = get_control_embed_model(self.output_maps, control_size)
        self.normalizer = layers.Normalization()
        self.network = get_control_network(self.input_map.shape, widths, block_depth, conv_x, control_size,
                                           attention_start, attention_heads, attention_modulo, condition_strategy)
        self.ema_network = keras.models.clone_model(self.network)
        self.use_sigmoid_loss = diffusion_loss == 'sigmoid'
        self.beta = sigmoid_beta
        self.supervisor = supervisor
        self.supervision_scalar = supervision_scalar
        self.inspect_model = False  # inspect_model

    def get_config(self):
        config = super().get_config().copy()
        config.update({'sigmoid_beta': self.beta, 'batch_size': self.batch_size})
        return config

    def compile(self, **kwargs):
        super().compile(**kwargs)
        self.noise_loss_tracker = keras.metrics.Mean(name="n_loss")
        self.image_loss_tracker = keras.metrics.Mean(name="i_loss")
        self.mse_metric = tf.keras.metrics.MeanSquaredError(name="mse")
        self.mae_metric = tf.keras.metrics.MeanAbsoluteError(name="mae")
        if self.supervisor is not None:
            self.supervised_loss_tracker = keras.metrics.Mean(name="supervised_loss")
        if self.input_map.axes() == 3 and self.inspect_model:
            self.kid = KernelInceptionDistance(name="kid", input_shape=self.input_map.shape, kernel_image_size=299)
            self.ms_ssim = MultiScaleSSIM()

    @property
    def metrics(self):
        m = [self.noise_loss_tracker, self.image_loss_tracker, self.mse_metric, self.mae_metric]
        if self.supervisor is not None:
            m.append(self.supervised_loss_tracker)
        if self.input_map.axes() == 3 and self.inspect_model:
            m.append(self.kid)
            m.append(self.ms_ssim)
        return m

    def denormalize(self, images):
        with tf.init_scope():
            mean = self.normalizer.mean
            var = self.normalizer.variance
        std = tf.sqrt(var)
        return images * std + mean

    def diffusion_schedule(self, diffusion_times):
        # diffusion times -> angles
        start_angle = tf.acos(max_signal_rate)
        end_angle = tf.acos(min_signal_rate)

        diffusion_angles = start_angle + diffusion_times * (end_angle - start_angle)

        # angles -> signal and noise rates
        signal_rates = tf.cos(diffusion_angles)
        noise_rates = tf.sin(diffusion_angles)
        # note that their squared sum is always: sin^2(x) + cos^2(x) = 1

        return noise_rates, signal_rates

    def denoise(self, control_embed, noisy_images, noise_rates, signal_rates, training):
        # the exponential moving average weights are used at evaluation
        if training:
            network = self.network
        else:
            network = self.ema_network

        # predict noise component and calculate the image component using it
        pred_noises = network([noisy_images, noise_rates ** 2, control_embed], training=training)
        pred_images = (noisy_images - noise_rates * pred_noises) / signal_rates

        return pred_noises, pred_images

    def reverse_diffusion(self, control_embed, initial_noise, diffusion_steps):
        # reverse diffusion = sampling
        num_images = initial_noise.shape[0]
        step_size = 1.0 / diffusion_steps

        # important line:
        # at the first sampling step, the "noisy image" is pure noise
        # but its signal rate is assumed to be nonzero (min_signal_rate)
        next_noisy_images = initial_noise
        pred_images = None
        for step in range(diffusion_steps):
            noisy_images = next_noisy_images

            # separate the current noisy image to its components
            diffusion_times = tf.ones([num_images] + [1] * self.input_map.axes()) - step * step_size
            noise_rates, signal_rates = self.diffusion_schedule(diffusion_times)
            pred_noises, pred_images = self.denoise(
                control_embed, noisy_images, noise_rates, signal_rates, training=False,
            )
            # network used in eval mode

            # remix the predicted components using the next signal and noise rates
            next_diffusion_times = diffusion_times - step_size
            next_noise_rates, next_signal_rates = self.diffusion_schedule(
                next_diffusion_times,
            )
            next_noisy_images = (
                    next_signal_rates * pred_images + next_noise_rates * pred_noises
            )
            # this new noisy image will be used in the next step
        return pred_images

    def generate(self, control_embed, num_images, diffusion_steps, reseed=None, renoise=None):
        # noise -> images -> denormalized images

        if reseed is not None:
            tf.random.set_seed(reseed)

        initial_noise = tf.random.normal(shape=(num_images,) + self.input_map.shape)

        generated_images = self.reverse_diffusion(control_embed, initial_noise, diffusion_steps)
        generated_images = self.denormalize(generated_images)
        return generated_images

    def generate_from_noise(self, control_embed, num_images, diffusion_steps, initial_noise):
        generated_images = self.reverse_diffusion(control_embed, initial_noise, diffusion_steps)
        generated_images = self.denormalize(generated_images)
        return generated_images

    def train_step(self, batch):
        # normalize images to have standard deviation of 1, like the noises
        images = batch[0][self.input_map.input_name()]
        # self.normalizer.adapt(images)
        images = self.normalizer(images, training=True)

        if self.autoencoder_control:
            control_embed = self.control_embed_model(batch[0])
        else:
            control_embed = self.control_embed_model(batch[1])

        noises = tf.random.normal(shape=(self.batch_size,) + self.input_map.shape)

        # sample uniform random diffusion times
        diffusion_times = tf.random.uniform(
            shape=[self.batch_size] + [1] * self.input_map.axes(), minval=0.0, maxval=1.0,
        )
        noise_rates, signal_rates = self.diffusion_schedule(diffusion_times)
        # mix the images with noises accordingly
        print(f'noises.shape {noises.shape} images.shape {images.shape}')
        noisy_images = signal_rates * images + noise_rates * noises

        with tf.GradientTape(persistent=True) if self.supervisor else tf.GradientTape() as tape:
            # train the network to separate noisy images to their components
            pred_noises, pred_images = self.denoise(
                control_embed, noisy_images, noise_rates, signal_rates, training=True,
            )

            noise_loss = self.loss(noises, pred_noises)  # used for training
            image_loss = self.loss(images, pred_images)  # only used as metric
            if self.use_sigmoid_loss:
                signal_rates_squared = tf.square(signal_rates)
                noise_rates_squared = tf.square(noise_rates)

                # Compute log-SNR (lambda_t)
                lambda_t = tf.math.log(signal_rates_squared / noise_rates_squared)
                weight = tf.math.sigmoid(self.beta - lambda_t)
                noise_loss = weight * noise_loss
            if self.supervisor is not None:
                if self.output_maps[0].is_categorical():
                    loss_fn = tf.keras.losses.CategoricalCrossentropy()
                else:
                    loss_fn = tf.keras.losses.MeanSquaredError()
                supervised_preds = self.supervisor(pred_images, training=True)
                supervised_loss = loss_fn(batch[1][self.output_maps[0].output_name()], supervised_preds)
                self.supervised_loss_tracker.update_state(supervised_loss)
                # Combine losses: add noise_loss and supervised_loss
                noise_loss += self.supervision_scalar * supervised_loss

                # Gradients for self.supervised_model
                supervised_gradients = tape.gradient(supervised_loss, self.supervisor.trainable_weights)
                self.optimizer.apply_gradients(zip(supervised_gradients, self.supervisor.trainable_weights))

        gradients = tape.gradient(noise_loss, self.network.trainable_weights)
        self.optimizer.apply_gradients(zip(gradients, self.network.trainable_weights))

        self.noise_loss_tracker.update_state(noise_loss)
        self.image_loss_tracker.update_state(image_loss)
        self.mse_metric.update_state(noises, pred_noises)
        self.mae_metric.update_state(noises, pred_noises)

        # track the exponential moving averages of weights
        for weight, ema_weight in zip(self.network.weights, self.ema_network.weights):
            ema_weight.assign(ema * ema_weight + (1 - ema) * weight)

        # KID is not measured during the training phase for computational efficiency
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, batch):
        # normalize images to have standard deviation of 1, like the noises
        images = batch[0][self.input_map.input_name()]
        # self.normalizer.adapt(images)
        images = self.normalizer(images, training=False)

        if self.autoencoder_control:
            control_embed = self.control_embed_model(batch[0])
        else:
            control_embed = self.control_embed_model(batch[1])

        noises = tf.random.normal(shape=(self.batch_size,) + self.input_map.shape)

        # sample uniform random diffusion times
        diffusion_times = tf.random.uniform(
            shape=[self.batch_size] + [1] * self.input_map.axes(), minval=0.0, maxval=1.0,
        )
        noise_rates, signal_rates = self.diffusion_schedule(diffusion_times)
        # mix the images with noises accordingly
        noisy_images = signal_rates * images + noise_rates * noises

        # use the network to separate noisy images to their components
        pred_noises, pred_images = self.denoise(
            control_embed, noisy_images, noise_rates, signal_rates, training=False,
        )

        noise_loss = self.loss(noises, pred_noises)
        image_loss = self.loss(images, pred_images)
        if self.use_sigmoid_loss:
            signal_rates_squared = tf.square(signal_rates)
            noise_rates_squared = tf.square(noise_rates)

            # Compute log-SNR (lambda_t)
            lambda_t = tf.math.log(signal_rates_squared / noise_rates_squared)
            weight = tf.math.sigmoid(self.beta - lambda_t)
            noise_loss = weight * noise_loss
        if self.supervisor is not None:
            if self.output_maps[0].is_categorical():
                loss_fn = tf.keras.losses.CategoricalCrossentropy()
            else:
                loss_fn = tf.keras.losses.MeanSquaredError()
            supervised_preds = self.supervisor(pred_images, training=True)
            supervised_loss = loss_fn(batch[1][self.output_maps[0].output_name()], supervised_preds)
            self.supervised_loss_tracker.update_state(supervised_loss)
            # Combine losses: add noise_loss and supervised_loss
            noise_loss += self.supervision_scalar * supervised_loss

        self.image_loss_tracker.update_state(image_loss)
        self.noise_loss_tracker.update_state(noise_loss)
        self.mse_metric.update_state(noises, pred_noises)
        self.mae_metric.update_state(noises, pred_noises)

        # measure KID between real and generated images
        # this is computationally demanding, kid_diffusion_steps has to be small
        if self.input_map.axes() == 3 and self.inspect_model:
            images = self.denormalize(images)
            generated_images = self.generate(control_embed,
                                             num_images=self.batch_size, diffusion_steps=20
                                             )
            self.kid.update_state(images, generated_images)
            self.ms_ssim.update_state(images, generated_images, 255)

        return {m.name: m.result() for m in self.metrics}

    def call(self, batch, training=False):
        """
        A minimal forward pass so that:
          1. Keras knows how to build() the model
          2. You can use model((noisy_images, noise_rates)) for inference
        """
        noisy_images, noise_rates = batch[0]
        if self.autoencoder_control:
            control_embed = self.control_embed_model(noisy_images)
        else:
            control_embed = self.control_embed_model(batch[1])
        # re-compute signal_rates
        signal_rates = tf.sqrt(1.0 - tf.square(noise_rates))
        # this returns (pred_noises, pred_images)
        return self.denoise(control_embed, noisy_images, noise_rates, signal_rates, training=training)

    def plot_images(self, epoch=None, logs=None, num_rows=1, num_cols=4, reseed=None, prefix='./figures/'):
        control_batch = {}
        for cm in self.output_maps:
            control_batch[cm.output_name()] = np.zeros((max(self.batch_size, num_rows * num_cols),) + cm.shape)
            if 'Sex' in cm.name:
                control_batch[cm.output_name()][:, 0] = 1  # all female

        print(f'\nControl batch keys: {list(control_batch.keys())}')
        control_embed = self.control_embed_model(control_batch)
        # plot random generated images for visual evaluation of generation quality
        generated_images = self.generate(
            control_embed,
            num_images=max(self.batch_size, num_rows * num_cols),
            diffusion_steps=plot_diffusion_steps,
            reseed=reseed,
        )
        plt.figure(figsize=(num_cols * 2.0, num_rows * 2.0), dpi=300)
        for row in range(num_rows):
            for col in range(num_cols):
                index = row * num_cols + col
                plt.subplot(num_rows, num_cols, index + 1)
                if generated_images.shape[-1] == 1:
                    plt.imshow(generated_images[index], cmap='gray')
                else:
                    img = generated_images[index].numpy()
                    img = (img - img.min()) / (1e-6 + img.max() - img.min())
                    plt.imshow(img)
                plt.axis("off")
        plt.tight_layout()
        now_string = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')
        figure_path = os.path.join(prefix, f'diffusion_image_generations_{now_string}{IMAGE_EXT}')
        if not os.path.exists(os.path.dirname(figure_path)):
            os.makedirs(os.path.dirname(figure_path))
        plt.savefig(figure_path, bbox_inches="tight")
        plt.close()

    def plot_reconstructions(
            self, batch, diffusion_amount=0,
            epoch=None, logs=None, num_rows=4, num_cols=4, prefix='./figures/',
    ):
        images = batch[0][self.input_map.input_name()]
        self.normalizer.adapt(images)
        images = self.normalizer(images, training=False)
        noises = tf.random.normal(shape=(self.batch_size,) + self.input_map.shape)
        diffusion_times = diffusion_amount * tf.ones(shape=[self.batch_size] + [1] * self.input_map.axes())
        noise_rates, signal_rates = self.diffusion_schedule(diffusion_times)
        # mix the images with noises accordingly
        noisy_images = signal_rates * images + noise_rates * noises

        if self.autoencoder_control:
            control_embed = self.control_embed_model(batch[0])
        else:
            control_embed = self.control_embed_model(batch[1])

        # use the network to separate noisy images to their components
        pred_noises, generated_images = self.denoise(
            control_embed, noisy_images, noise_rates, signal_rates, training=False,
        )
        plt.figure(figsize=(num_cols * 2.0, num_rows * 2.0), dpi=300)
        for row in range(num_rows):
            for col in range(num_cols):
                index = row * num_cols + col
                plt.subplot(num_rows, num_cols, index + 1)
                if generated_images.shape[-1] == 1:
                    plt.imshow(generated_images[index], cmap='gray')
                else:
                    img = generated_images[index].numpy()
                    img = (img - img.min()) / (1e-6 + img.max() - img.min())
                    plt.imshow(img)
                plt.axis("off")
        plt.tight_layout()
        now_string = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')
        figure_path = os.path.join(prefix, f'diffusion_image_reconstructions_{now_string}{IMAGE_EXT}')
        if not os.path.exists(os.path.dirname(figure_path)):
            os.makedirs(os.path.dirname(figure_path))
        plt.savefig(figure_path, bbox_inches="tight")
        plt.close()
        plt.figure(figsize=(num_cols * 2.0, num_rows * 2.0), dpi=300)
        for row in range(num_rows):
            for col in range(num_cols):
                index = row * num_cols + col
                plt.subplot(num_rows, num_cols, index + 1)
                if images.shape[-1] == 1:
                    plt.imshow(images[index], cmap='gray')
                else:
                    img = images[index].numpy()
                    img = (img - img.min()) / (1e-6 + img.max() - img.min())
                    plt.imshow(img)
                plt.axis("off")
        plt.tight_layout()
        now_string = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')
        figure_path = os.path.join(prefix, f'input_images_{now_string}{IMAGE_EXT}')
        if not os.path.exists(os.path.dirname(figure_path)):
            os.makedirs(os.path.dirname(figure_path))
        plt.savefig(figure_path, bbox_inches="tight")
        plt.close()
        return generated_images

    def control_plot_images(
            self, control_batch, epoch=None, logs=None, num_rows=2, num_cols=8, reseed=None,
            renoise=None,
    ):
        control_embed = self.control_embed_model(control_batch)
        # plot random generated images for visual evaluation of generation quality
        generated_images = self.generate(
            control_embed,
            num_images=max(self.batch_size, num_rows * num_cols),
            diffusion_steps=plot_diffusion_steps,
            reseed=reseed,
            renoise=renoise,
        )

        plt.figure(figsize=(num_cols * 2.0, num_rows * 2.0), dpi=300)
        for row in range(num_rows):
            for col in range(num_cols):
                index = row * num_cols + col
                plt.subplot(num_rows, num_cols, index + 1)
                if generated_images.shape[-1] == 1:
                    plt.imshow(generated_images[index], cmap='gray')
                else:
                    img = generated_images[index].numpy()
                    img = (img - img.min()) / (1e-6 + img.max() - img.min())
                    plt.imshow(img)
                plt.axis("off")
        plt.tight_layout()
        plt.show()
        plt.close()

        return generated_images

    def control_plot_images_embed(
            self, control_embed, epoch=None, logs=None, num_rows=2, num_cols=8, reseed=None,
            renoise=None,
    ):
        generated_images = self.generate(
            control_embed,
            num_images=max(self.batch_size, num_rows * num_cols),
            diffusion_steps=plot_diffusion_steps,
            reseed=reseed,
            renoise=renoise,
        )

        plt.figure(figsize=(num_cols * 2.0, num_rows * 2.0), dpi=300)
        for row in range(num_rows):
            for col in range(num_cols):
                index = row * num_cols + col
                plt.subplot(num_rows, num_cols, index + 1)
                if generated_images.shape[-1] == 1:
                    plt.imshow(generated_images[index], cmap='gray')
                else:
                    img = generated_images[index].numpy()
                    img = (img - img.min()) / (1e-6 + img.max() - img.min())
                    plt.imshow(img)
                plt.axis("off")
        plt.tight_layout()
        plt.show()
        plt.close()

        return generated_images

    def control_plot_images_noise(self, control_batch, initial_noise, epoch=None, logs=None, num_rows=2, num_cols=8):
        control_embed = self.control_embed_model(control_batch)
        # plot random generated images for visual evaluation of generation quality
        generated_images = self.generate_from_noise(
            control_embed,
            num_images=max(self.batch_size, num_rows * num_cols),
            diffusion_steps=plot_diffusion_steps,
            initial_noise=initial_noise,
        )

        plt.figure(figsize=(num_cols * 2.0, num_rows * 2.0), dpi=300)
        for row in range(num_rows):
            for col in range(num_cols):
                index = row * num_cols + col
                plt.subplot(num_rows, num_cols, index + 1)
                if generated_images.shape[-1] == 1:
                    plt.imshow(generated_images[index], cmap='gray')
                else:
                    img = generated_images[index].numpy()
                    img = (img - img.min()) / (1e-6 + img.max() - img.min())
                    plt.imshow(img)
                plt.axis("off")
        plt.tight_layout()
        plt.show()
        plt.close()

        return generated_images

    def control_plot_ecgs(
            self, control_batch, epoch=None, logs=None, num_rows=2, num_cols=8, reseed=None,
            renoise=None,
    ):
        control_embed = self.control_embed_model(control_batch)
        # plot random generated images for visual evaluation of generation quality
        generated_images = self.generate(
            control_embed,
            num_images=max(self.batch_size, num_rows * num_cols),
            diffusion_steps=plot_diffusion_steps,
            reseed=reseed,
            renoise=renoise,
        )

        plt.figure(figsize=(num_cols * 2.0, num_rows * 2.0), dpi=300)
        for row in range(num_rows):
            for col in range(num_cols):
                index = row * num_cols + col
                plt.subplot(num_rows, num_cols, index + 1)
                for lead in range(generated_images.shape[-1]):
                    plt.plot(generated_images[index, :, lead], label=lead)
                plt.axis("off")
        plt.tight_layout()
        plt.show()
        plt.close()

        return generated_images

    def plot_ecgs(self, epoch=None, logs=None, num_rows=1, num_cols=4, reseed=None, prefix='./figures/'):
        control_batch = {}
        for cm in self.output_maps:
            control_batch[cm.output_name()] = np.zeros((max(self.batch_size, num_rows * num_cols),) + cm.shape)
            if 'Sex' in cm.name:
                control_batch[cm.output_name()][:, 0] = 1  # all female

        print(f'\nControl batch keys: {list(control_batch.keys())}')
        control_embed = self.control_embed_model(control_batch)

        # plot random generated images for visual evaluation of generation quality
        generated_images = self.generate(
            control_embed,
            num_images=max(self.batch_size, num_rows * num_cols),
            diffusion_steps=plot_diffusion_steps,
            reseed=reseed,
        )
        logging.info(f'Generated ECGs with shape:{generated_images.shape}')
        plt.figure(figsize=(num_cols * 2.0, num_rows * 2.0), dpi=300)
        for row in range(num_rows):
            for col in range(num_cols):
                index = row * num_cols + col
                plt.subplot(num_rows, num_cols, index + 1)
                for lead in range(generated_images.shape[-1]):
                    plt.plot(generated_images[index, :, lead], label=lead)
                plt.axis("off")
        plt.tight_layout()
        now_string = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')
        figure_path = os.path.join(prefix, f'diffusion_ecg_generations_{now_string}{IMAGE_EXT}')
        if not os.path.exists(os.path.dirname(figure_path)):
            os.makedirs(os.path.dirname(figure_path))
        plt.savefig(figure_path, bbox_inches="tight")
        plt.close()
