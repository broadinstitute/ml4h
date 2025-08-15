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
min_signal_rate = 0.05
max_signal_rate = 0.95

# architecture
embedding_dims = 256
embedding_max_frequency = 1000.0

# optimization
ema = 0.999

# plotting
plot_diffusion_steps = 50

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

@register_keras_serializable()
def residual_block(width, conv, kernel_size, groups=32):
    def apply(x):
        # shortcut
        input_channels = x.shape[-1]
        if input_channels != width:
            residual = conv(width, kernel_size=1)(x)
        else:
            residual = x

        # first GN → SiLU → Conv
        x = tf.keras.layers.GroupNormalization(groups=groups, axis=-1)(x)
        x = layers.Activation('silu')(x)
        x = conv(width, kernel_size=kernel_size, padding="same")(x)

        # second GN → SiLU → Conv
        x = tf.keras.layers.GroupNormalization(groups=groups, axis=-1)(x)
        x = layers.Activation('silu')(x)
        x = conv(width, kernel_size=kernel_size, padding="same")(x)

        # merge
        x = layers.Add()([x, residual])
        return x
    return apply

@register_keras_serializable()
def down_block(width, block_depth, conv, pool, kernel_size, groups=32):
    def apply(x):
        x, skips = x
        for _ in range(block_depth):
            x = residual_block(width, conv, kernel_size, groups=groups)(x)
            skips.append(x)
        x = pool(pool_size=2)(x)
        return x
    return apply

@register_keras_serializable()
def up_block(width, block_depth, conv, upsample, kernel_size, groups=32):
    def apply(x):
        x, skips = x
        x = upsample(size=2)(x)
        for _ in range(block_depth):
            x = layers.Concatenate()([x, skips.pop()])
            x = residual_block(width, conv, kernel_size, groups=groups)(x)
        return x
    return apply

@register_keras_serializable()
def get_network(input_shape, widths, block_depth, kernel_size):
    noisy_images = keras.Input(shape=input_shape)
    conv, upsample, pool, _ = layers_from_shape_control(input_shape)
    noise_variances = keras.Input(shape=[1] * len(input_shape))

    e = layers.Lambda(sinusoidal_embedding)(noise_variances)
    if len(input_shape) == 2:
        e = upsample(size=input_shape[-2])(e)
    else:
        e = upsample(size=input_shape[:-1], interpolation="nearest")(e)

    print(f'e shape: {e.shape} len {len(input_shape)}')
    x = conv(widths[0], kernel_size=1)(noisy_images)
    x = layers.Concatenate()([x, e])

    skips = []
    for width in widths[:-1]:
        x = down_block(width, block_depth, conv, pool, kernel_size)([x, skips])

    for _ in range(block_depth):
        x = residual_block(widths[-1], conv, kernel_size)(x)

    for width in reversed(widths[:-1]):
        x = up_block(width, block_depth, conv, upsample, kernel_size)([x, skips])

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
                 name=None,
                 **kwargs):
        super().__init__()

        self.tensor_map     = tensor_map
        self.batch_size     = batch_size
        self.widths         = widths
        self.block_depth    = block_depth
        self.kernel_size    = kernel_size
        self.diffusion_loss = diffusion_loss
        self.sigmoid_beta   = sigmoid_beta
        self.inspect_model  = False #inspect_model

        self.normalizer = layers.Normalization()
        self.network = get_network(self.tensor_map.shape, widths, block_depth, kernel_size)
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
        # the exponential moving average weights are used at evaluation
        if training:
            network = self.network
        else:
            network = self.ema_network

        # predict noise component and calculate the image component using it
        pred_noises = network([noisy_images, noise_rates ** 2], training=training)
        pred_images = (noisy_images - noise_rates * pred_noises) / signal_rates

        return pred_noises, pred_images

    def reverse_diffusion(self, initial_noise, diffusion_steps):
        # reverse diffusion = sampling
        num_images = initial_noise.shape[0]
        step_size = 1.0 / diffusion_steps

        # important line:
        # at the first sampling step, the "noisy image" is pure noise
        # but its signal rate is assumed to be nonzero (min_signal_rate)
        next_noisy_images = initial_noise
        for step in range(diffusion_steps):
            noisy_images = next_noisy_images

            # separate the current noisy image to its components
            diffusion_times = tf.ones([num_images] + [1] * self.tensor_map.axes()) - step * step_size
            noise_rates, signal_rates = self.diffusion_schedule(diffusion_times)
            pred_noises, pred_images = self.denoise(
                noisy_images, noise_rates, signal_rates, training=False,
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

    def generate(self, num_images, diffusion_steps, reseed=None):
        # noise -> images -> denormalized images
        if reseed is not None:
            tf.random.set_seed(reseed)
        initial_noise = tf.random.normal(shape=(num_images,) + self.tensor_map.shape)
        generated_images = self.reverse_diffusion(initial_noise, diffusion_steps)
        generated_images = self.denormalize(generated_images)
        return generated_images

    def train_step(self, images_original):
        # normalize images to have standard deviation of 1, like the noises
        images = images_original[0][self.tensor_map.input_name()]
        #self.normalizer.adapt(images)
        # images = images['input_lax_4ch_diastole_slice0_224_3d_continuous']
        images = self.normalizer(images, training=True)
        # images = images.numpy() - images.numpy().mean() / images.numpy().std()
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
            # train the network to separate noisy images to their components
            pred_noises, pred_images = self.denoise(
                noisy_images, noise_rates, signal_rates, training=True,
            )

            noise_loss = self.loss(noises, pred_noises)  # used for training
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
        #self.normalizer.adapt(images)
        images = self.normalizer(images, training=False)
        # images = images - tf.math.reduce_mean(images) / tf.math.reduce_std(images)
        noises = tf.random.normal(shape=(self.batch_size,) + self.tensor_map.shape)

        # sample uniform random diffusion times
        diffusion_times = tf.random.uniform(
            shape=[self.batch_size] + [1] * self.tensor_map.axes(), minval=0.0, maxval=1.0,
        )
        noise_rates, signal_rates = self.diffusion_schedule(diffusion_times)
        # mix the images with noises accordingly
        noisy_images = signal_rates * images + noise_rates * noises

        # use the network to separate noisy images to their components
        pred_noises, pred_images = self.denoise(
            noisy_images, noise_rates, signal_rates, training=False,
        )

        noise_loss = self.loss(noises, pred_noises)
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
                num_images=self.batch_size, diffusion_steps=20
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
                if len(generated_images[index].shape) == 3 and generated_images[index].shape[-1] > 1:
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
                plt.imshow(generated_images[index], cmap='gray')
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
                plt.imshow(images[index], cmap='gray')
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
            self.control_embed_model = load_model(encoder_file, compile=False)
            logging.info(f'loaded encoder for DiffAE at: {encoder_file}')
        else:
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

        control_embed = self.control_embed_model(batch[0])

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

        control_embed = self.control_embed_model(batch[0])

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
        control_embed = self.control_embed_model(noisy_images)
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
                plt.imshow(generated_images[index], cmap='gray')
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

        control_embed = self.control_embed_model(batch[0])

        # use the network to separate noisy images to their components
        pred_noises, generated_images = self.denoise(
            control_embed, noisy_images, noise_rates, signal_rates, training=False,
        )
        plt.figure(figsize=(num_cols * 2.0, num_rows * 2.0), dpi=300)
        for row in range(num_rows):
            for col in range(num_cols):
                index = row * num_cols + col
                plt.subplot(num_rows, num_cols, index + 1)
                plt.imshow(generated_images[index], cmap='gray')
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
                plt.imshow(images[index], cmap='gray')
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
                plt.imshow(generated_images[index], cmap='gray')
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
                plt.imshow(generated_images[index], cmap='gray')
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
                plt.imshow(generated_images[index], cmap='gray')
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
