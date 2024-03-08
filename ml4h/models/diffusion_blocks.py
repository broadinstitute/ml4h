import logging
import os
from typing import Dict, List, Tuple, Sequence
import math

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow import keras
from keras import layers

from ml4h.defines import IMAGE_EXT
from ml4h.models.Block import Block
from ml4h.TensorMap import TensorMap


Tensor = tf.Tensor

# sampling
min_signal_rate = 0.02
max_signal_rate = 0.95

# architecture
embedding_dims = 256
embedding_max_frequency = 1000.0

# optimization
batch_size = 4
ema = 0.999
learning_rate = 5e-4
weight_decay = 1e-4

# plotting
plot_diffusion_steps = 20


def sinusoidal_embedding(x):
    embedding_min_frequency = 1.0
    frequencies = tf.exp(
        tf.linspace(
            tf.math.log(embedding_min_frequency),
            tf.math.log(embedding_max_frequency),
            embedding_dims // 2,
        )
    )
    angular_speeds = 2.0 * math.pi * frequencies
    #     embeddings = tf.concat(
    #         [tf.sin(angular_speeds * x), tf.cos(angular_speeds * x)], axis=3
    #     )
    embeddings = tf.concat(
        [tf.sin(angular_speeds * x)], axis=2
    )
    return embeddings


def ResidualBlock(width, conv, kernel_size):
    def apply(x):
        input_width = x.shape[-1]
        if input_width == width:
            residual = x
        else:
            residual = conv(width, kernel_size=1)(x)
        x = layers.BatchNormalization(center=False, scale=False)(x)
        x = conv(
            width, kernel_size=kernel_size, padding="same", activation=keras.activations.swish
        )(x)
        x = conv(width, kernel_size=kernel_size, padding="same")(x)
        x = layers.Add()([x, residual])
        return x

    return apply


def DownBlock(width, block_depth, conv, pool, kernel_size):
    def apply(x):
        x, skips = x
        for _ in range(block_depth):
            x = ResidualBlock(width, conv,  kernel_size)(x)
            skips.append(x)
        x = pool(pool_size=2)(x)
        return x

    return apply


def UpBlock(width, block_depth, conv, upsample, kernel_size):
    def apply(x):
        x, skips = x
        # x = upsample(size=2, interpolation="bilinear")(x)
        x = upsample(size=2)(x)
        for _ in range(block_depth):
            x = layers.Concatenate()([x, skips.pop()])
            x = ResidualBlock(width, conv, kernel_size)(x)
        return x

    return apply


def layers_from_shape(input_shape):
    if len(input_shape) == 2:
        return layers.Conv1D, layers.UpSampling1D, layers.AveragePooling1D
    elif len(input_shape) == 3:
        return layers.Conv2D, layers.UpSampling2D, layers.AveragePooling2D
    elif len(input_shape) == 4:
        return layers.Conv3D, layers.UpSampling3D, layers.AveragePooling3D


def get_network(input_shape, widths, block_depth, kernel_size):
    noisy_images = keras.Input(shape=input_shape)
    conv, upsample, pool = layers_from_shape(input_shape)
    noise_variances = keras.Input(shape=[1] * len(input_shape))

    e = layers.Lambda(sinusoidal_embedding)(noise_variances)
    # e = upsample(size=input_shape[:-1], interpolation="nearest")(e)
    e = upsample(size=input_shape[-2])(e)
    print(f'e shape: {e.shape}')
    x = conv(widths[0], kernel_size=1)(noisy_images)
    x = layers.Concatenate()([x, e])

    skips = []
    for width in widths[:-1]:
        x = DownBlock(width, block_depth, conv, pool, kernel_size)([x, skips])

    for _ in range(block_depth):
        x = ResidualBlock(widths[-1], conv, kernel_size)(x)

    for width in reversed(widths[:-1]):
        x = UpBlock(width, block_depth, conv, upsample, kernel_size)([x, skips])

    x = conv(input_shape[-1], kernel_size=1, kernel_initializer="zeros")(x)

    return keras.Model([noisy_images, noise_variances], x, name="residual_unet")


class DiffusionModel(keras.Model):
    def __init__(self, tensor_map, batch_size, widths, block_depth, kernel_size):
        super().__init__()

        self.tensor_map = tensor_map
        self.batch_size = batch_size
        self.normalizer = layers.Normalization()
        self.network = get_network(self.tensor_map.shape, widths, block_depth, kernel_size)
        self.ema_network = keras.models.clone_model(self.network)

    def can_apply(self):
        return self.tensor_map.axes() > 1

    def compile(self, **kwargs):
        super().compile(**kwargs)

        self.noise_loss_tracker = keras.metrics.Mean(name="n_loss")
        self.image_loss_tracker = keras.metrics.Mean(name="i_loss")
        # self.kid = KID(name = "kid", input_shape = self.tensor_map.shape)

    @property
    def metrics(self):
        return [self.noise_loss_tracker, self.image_loss_tracker]

    def denormalize(self, images):
        # convert the pixel values back to 0-1 range
        # images = images - tf.math.reduce_mean(images) + images * tf.math.reduce_std(images)
        images = self.normalizer.mean + images * self.normalizer.variance ** 0.5
        # print(f'images max min {images}')
        return images  # tf.clip_by_value(images, 0.0, 1.0)

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
            diffusion_times = tf.ones([num_images, ] + [1] * self.tensor_map.axes()) - step * step_size
            noise_rates, signal_rates = self.diffusion_schedule(diffusion_times)
            pred_noises, pred_images = self.denoise(
                noisy_images, noise_rates, signal_rates, training=False
            )
            # network used in eval mode

            # remix the predicted components using the next signal and noise rates
            next_diffusion_times = diffusion_times - step_size
            next_noise_rates, next_signal_rates = self.diffusion_schedule(
                next_diffusion_times
            )
            next_noisy_images = (
                    next_signal_rates * pred_images + next_noise_rates * pred_noises
            )
            # this new noisy image will be used in the next step

        return pred_images

    def generate(self, num_images, diffusion_steps):
        # noise -> images -> denormalized images
        initial_noise = tf.random.normal(shape=(num_images,) + self.tensor_map.shape)
        generated_images = self.reverse_diffusion(initial_noise, diffusion_steps)
        generated_images = self.denormalize(generated_images)
        return generated_images

    def train_step(self, images_original):
        # normalize images to have standard deviation of 1, like the noises
        images = images_original[0][self.tensor_map.input_name()]
        self.normalizer.update_state(images)
        # images = images['input_lax_4ch_diastole_slice0_224_3d_continuous']
        images = self.normalizer(images, training=True)
        # images = images.numpy() - images.numpy().mean() / images.numpy().std()
        noises = tf.random.normal(shape=(batch_size,) + self.tensor_map.shape)

        # sample uniform random diffusion times
        diffusion_times = tf.random.uniform(
            shape=[self.batch_size, ] + [1] * self.tensor_map.axes(), minval=0.0, maxval=1.0
        )
        noise_rates, signal_rates = self.diffusion_schedule(diffusion_times)
        # mix the images with noises accordingly
        print(f'noises.shape {noises.shape} images.shape {images.shape}')
        noisy_images = signal_rates * images + noise_rates * noises

        with tf.GradientTape() as tape:
            # train the network to separate noisy images to their components
            pred_noises, pred_images = self.denoise(
                noisy_images, noise_rates, signal_rates, training=True
            )

            noise_loss = self.loss(noises, pred_noises)  # used for training
            image_loss = self.loss(images, pred_images)  # only used as metric

        gradients = tape.gradient(noise_loss, self.network.trainable_weights)
        self.optimizer.apply_gradients(zip(gradients, self.network.trainable_weights))

        self.noise_loss_tracker.update_state(noise_loss)
        self.image_loss_tracker.update_state(image_loss)

        # track the exponential moving averages of weights
        for weight, ema_weight in zip(self.network.weights, self.ema_network.weights):
            ema_weight.assign(ema * ema_weight + (1 - ema) * weight)

        # KID is not measured during the training phase for computational efficiency
        return {m.name: m.result() for m in self.metrics[:-1]}

    def test_step(self, images_original):
        # normalize images to have standard deviation of 1, like the noises
        images = images_original[0][self.tensor_map.input_name()]
        self.normalizer.update_state(images)
        images = self.normalizer(images, training=False)
        # images = images - tf.math.reduce_mean(images) / tf.math.reduce_std(images)
        noises = tf.random.normal(shape=(batch_size,) + self.tensor_map.shape)

        # sample uniform random diffusion times
        diffusion_times = tf.random.uniform(
            shape=[self.batch_size, ] + [1] * self.tensor_map.axes(), minval=0.0, maxval=1.0
        )
        noise_rates, signal_rates = self.diffusion_schedule(diffusion_times)
        # mix the images with noises accordingly
        noisy_images = signal_rates * images + noise_rates * noises

        # use the network to separate noisy images to their components
        pred_noises, pred_images = self.denoise(
            noisy_images, noise_rates, signal_rates, training=False
        )

        noise_loss = self.loss(noises, pred_noises)
        image_loss = self.loss(images, pred_images)

        self.image_loss_tracker.update_state(image_loss)
        self.noise_loss_tracker.update_state(noise_loss)

        # measure KID between real and generated images
        # this is computationally demanding, kid_diffusion_steps has to be small
        images = self.denormalize(images)
        generated_images = self.generate(
            num_images=batch_size, diffusion_steps=20
        )
        #         self.kid.update_state(images, generated_images)

        return {m.name: m.result() for m in self.metrics}

    def plot_images(self, epoch=None, logs=None, num_rows=3, num_cols=6, prefix='./figures/'):
        # plot random generated images for visual evaluation of generation quality
        generated_images = self.generate(
            num_images=num_rows * num_cols,
            diffusion_steps=plot_diffusion_steps,
        )

        plt.figure(figsize=(num_cols * 2.0, num_rows * 2.0), dpi=300)
        for row in range(num_rows):
            for col in range(num_cols):
                index = row * num_cols + col
                plt.subplot(num_rows, num_cols, index + 1)
                plt.imshow(generated_images[index], cmap='gray')
                plt.axis("off")
        plt.tight_layout()
        figure_path = os.path.join(prefix, "diffusion_generations" + IMAGE_EXT)
        if not os.path.exists(os.path.dirname(figure_path)):
            os.makedirs(os.path.dirname(figure_path))
        plt.savefig(figure_path, bbox_inches="tight")

    def plot_ecgs(self, epoch=None, logs=None, num_rows=2, num_cols=8, reseed=None, prefix='./figures/'):
        # plot random generated images for visual evaluation of generation quality
        generated_images = self.generate(
            num_images=max(batch_size, num_rows * num_cols),
            diffusion_steps=plot_diffusion_steps,
        )

        plt.figure(figsize=(num_cols * 2.0, num_rows * 2.0), dpi=300)
        for row in range(num_rows):
            for col in range(num_cols):
                index = row * num_cols + col
                plt.subplot(num_rows, num_cols, index + 1)
                plt.plot(generated_images[index, ..., 0])
                plt.axis("off")
        plt.tight_layout()
        figure_path = os.path.join(prefix, "diffusion_generations" + IMAGE_EXT)
        if not os.path.exists(os.path.dirname(figure_path)):
            os.makedirs(os.path.dirname(figure_path))
        plt.savefig(figure_path, bbox_inches="tight")

    def plot_reconstructions(self, images_original, diffusion_amount=0,
                             epoch=None, logs=None, num_rows=3, num_cols=6):
        images = images_original[0][self.tensor_map.input_name()]
        self.normalizer.update_state(images)
        images = self.normalizer(images, training=False)
        noises = tf.random.normal(shape=(batch_size,) + self.tensor_map.shape)

        diffusion_times = diffusion_amount * tf.ones(shape=[self.batch_size, ] + [1] * self.tensor_map.axes())
        noise_rates, signal_rates = self.diffusion_schedule(diffusion_times)
        # mix the images with noises accordingly
        noisy_images = signal_rates * images + noise_rates * noises

        # use the network to separate noisy images to their components
        pred_noises, generated_images = self.denoise(
            noisy_images, noise_rates, signal_rates, training=False
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

    def in_paint(self, images_original, masks, diffusion_steps=64, num_rows=3, num_cols=6):
        images = images_original[0][self.tensor_map.input_name()]
        noises = tf.random.normal(shape=(batch_size,) + self.tensor_map.shape)
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
            diffusion_times = tf.ones([num_images, ] + [1] * self.tensor_map.axes()) - step * step_size
            noise_rates, signal_rates = self.diffusion_schedule(diffusion_times)
            pred_noises, generated_images = self.denoise(
                noisy_images, noise_rates, signal_rates, training=False
            )

            # apply the mask
            generated_images = generated_images * (1 - masks) + images * masks

            # remix the predicted components using the next signal and noise rates
            next_diffusion_times = diffusion_times - step_size
            next_noise_rates, next_signal_rates = self.diffusion_schedule(
                next_diffusion_times
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


class DiffusionBlock(Block):
    def __init__(
            self,
            *,
            tensor_map: TensorMap,
            dense_blocks: List[int] = [32, 32, 32],
            dense_layers: List[int] = [256],
            batch_size: int = 16,
            block_size: int = 3,
            conv_x: int = 3,
            activation: str = 'swish',
            **kwargs,
    ):
        self.tensor_map = tensor_map
        self.batch_size = batch_size
        if not self.can_apply():
            return

        self.diffusion_model = DiffusionModel(tensor_map, dense_blocks, block_size, conv_x)
        import tensorflow_addons as tfa
        self.diffusion_model.compile(
            optimizer=tfa.optimizers.AdamW(
                learning_rate=learning_rate, weight_decay=weight_decay
            ),
            loss=keras.losses.mean_absolute_error,
        )
    def can_apply(self):
        return self.tensor_map.axes() > 1

    def __call__(self, x: Tensor, intermediates: Dict[TensorMap, List[Tensor]] = None) -> Tensor:
        if not self.can_apply():
            return x
        times = tf.ones([self.batch_size]+[1]*self.tensor_map.axes())
        x = self.diffusion_model([x, times])
        #x = self.loss_layer(x)
        intermediates[self.tensor_map].append(x)
        return x

