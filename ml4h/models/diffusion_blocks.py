import datetime
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
ema = 0.999
learning_rate = 5e-4
weight_decay = 1e-4

# plotting
plot_diffusion_steps = 20


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


def residual_block(width, conv, kernel_size):
    def apply(x):
        input_width = x.shape[-1]
        if input_width == width:
            residual = x
        else:
            residual = conv(width, kernel_size=1)(x)
        x = layers.BatchNormalization(center=False, scale=False)(x)
        x = conv(
            width, kernel_size=kernel_size, padding="same", activation=keras.activations.swish,
        )(x)
        x = conv(width, kernel_size=kernel_size, padding="same")(x)
        x = layers.Add()([x, residual])
        return x

    return apply


def down_block(width, block_depth, conv, pool, kernel_size):
    def apply(x):
        x, skips = x
        for _ in range(block_depth):
            x = residual_block(width, conv, kernel_size)(x)
            skips.append(x)
        x = pool(pool_size=2)(x)
        return x

    return apply


def up_block(width, block_depth, conv, upsample, kernel_size):
    def apply(x):
        x, skips = x
        # x = upsample(size=2, interpolation="bilinear")(x)
        x = upsample(size=2)(x)
        for _ in range(block_depth):
            x = layers.Concatenate()([x, skips.pop()])
            x = residual_block(width, conv, kernel_size)(x)
        return x

    return apply


def residual_block_control(width, conv, kernel_size, attention_heads):
    def apply(x):
        x, control = x
        input_width = x.shape[-1]
        if input_width == width:
            residual = x
        else:
            residual = conv(width, kernel_size=1)(x)
        x = layers.BatchNormalization(center=False, scale=False)(x)
        x = conv(
            width, kernel_size=kernel_size, padding="same", activation=keras.activations.swish
        )(x)
        x = keras.layers.MultiHeadAttention(num_heads = attention_heads, key_dim = width)(x, control)
        x = conv(width, kernel_size=kernel_size, padding="same")(x)
        x = layers.Add()([x, residual])
        return x

    return apply


def down_block_control(width, block_depth, conv, pool, kernel_size, attention_heads):
    def apply(x):
        x, skips, control = x
        for _ in range(block_depth):
            x = residual_block_control(width, conv, kernel_size, attention_heads)([x, control])
            skips.append(x)
        x = pool(pool_size=2)(x)
        return x

    return apply


def up_block_control(width, block_depth, conv, upsample, kernel_size, attention_heads):
    def apply(x):
        x, skips, control = x
        # x = upsample(size=2, interpolation="bilinear")(x)
        x = upsample(size=2)(x)
        for _ in range(block_depth):
            x = layers.Concatenate()([x, skips.pop()])
            x = residual_block_control(width, conv, kernel_size, attention_heads)([x, control])
        return x

    return apply


def get_control_network(input_shape, widths, block_depth, kernel_size, control_size,
                        attention_start, attention_heads, attention_modulo):
    noisy_images = keras.Input(shape=input_shape)
    noise_variances = keras.Input(shape=[1] * len(input_shape))

    conv, upsample, pool, control_idxs = layers_from_shape_control(input_shape)
    control = keras.Input(shape=(control_size,))

    x = conv(widths[0], kernel_size=1)(noisy_images)
    e = layers.Lambda(sinusoidal_embedding)(noise_variances)

    if len(input_shape) == 2:  # 1D Signals
        e = upsample(size=input_shape[-2])(e)
        c = upsample(size=input_shape[-2])(control[control_idxs])
    else:
        e = upsample(size=input_shape[:-1], interpolation="nearest")(e)
        c = upsample(size=input_shape[:-1])(control[control_idxs])

    print(f'Control up-sampled shape shape: {c.shape} e shape {e.shape} x: {x.shape}')

    x = layers.Concatenate()([x, e])

    skips = []
    for i, width in enumerate(widths[:-1]):
        if attention_modulo > 1 and (i + 1) % attention_modulo == 0:
            if len(input_shape) > 2:
                c2 = upsample(size=x.shape[1:-1])(control[control_idxs])
            else:
                c2 = upsample(size=x.shape[-2])(control[control_idxs])
            x = down_block_control(width, block_depth, conv, pool, kernel_size, attention_heads)([x, skips, c2])
        else:
            x = down_block(width, block_depth, conv, pool, kernel_size)([x, skips])

    if len(input_shape) > 2:
        c2 = upsample(size=x.shape[1:-1])(control[control_idxs])
    else:
        c2 = upsample(size=x.shape[-2])(control[control_idxs])

    for i in range(block_depth):
        x = residual_block_control(widths[-1], conv, kernel_size, attention_heads)([x, c2])

    for i, width in enumerate(reversed(widths[:-1])):
        if attention_modulo > 1 and i % attention_modulo == 0:
            if len(input_shape) > 2:
                c2 = upsample(size=x.shape[1:-1])(control[control_idxs])
            else:
                c2 = upsample(size=x.shape[-2])(control[control_idxs])
            x = up_block_control(width, block_depth, conv, upsample, kernel_size, attention_heads)([x, skips, c2])
        else:
            x = up_block(width, block_depth, conv, upsample, kernel_size)([x, skips])

    x = conv(input_shape[-1], kernel_size=1, activation="linear", kernel_initializer="zeros")(x)

    return keras.Model([noisy_images, noise_variances, control], x, name="control_unet")


def get_control_embed_model(output_maps, control_size):
    control_ins = []
    logging.info(f'Build control embedder on :{output_maps}')
    for cm in output_maps:
        control_ins.append(keras.Input(shape=cm.shape, name=cm.output_name()))
    c = layers.Concatenate()(control_ins)
    #c = layers.Dense(control_size, activation='linear')(c)
    return keras.Model(control_ins, c, name='control_embed')


class DiffusionController(keras.Model):
    def __init__(
        self, tensor_map, output_maps, batch_size, widths, block_depth, conv_x, control_size,
        attention_start, attention_heads, attention_modulo, diffusion_loss, sigmoid_beta
    ):
        super().__init__()

        self.input_map = tensor_map
        self.batch_size = batch_size
        self.output_maps = output_maps
        self.control_embed_model = get_control_embed_model(self.output_maps, control_size)
        self.normalizer = layers.Normalization()
        self.network = get_control_network(self.input_map.shape, widths, block_depth, conv_x, control_size,
                                           attention_start, attention_heads, attention_modulo)
        self.ema_network = keras.models.clone_model(self.network)
        self.use_sigmoid_loss = diffusion_loss == 'sigmoid'
        self.beta = sigmoid_beta


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
        self.normalizer.update_state(images)
        images = self.normalizer(images, training=True)

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

        with tf.GradientTape() as tape:
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

        gradients = tape.gradient(noise_loss, self.network.trainable_weights)
        self.optimizer.apply_gradients(zip(gradients, self.network.trainable_weights))

        self.noise_loss_tracker.update_state(noise_loss)
        self.image_loss_tracker.update_state(image_loss)

        # track the exponential moving averages of weights
        for weight, ema_weight in zip(self.network.weights, self.ema_network.weights):
            ema_weight.assign(ema * ema_weight + (1 - ema) * weight)

        # KID is not measured during the training phase for computational efficiency
        return {m.name: m.result() for m in self.metrics[:-1]}

    # def call(self, inputs):
    #     # normalize images to have standard deviation of 1, like the noises
    #     images = inputs[self.input_map.input_name()]
    #     self.normalizer.update_state(images)
    #     images = self.normalizer(images, training=False)

    #     control_embed = self.control_embed_model(inputs)

    #     noises = tf.random.normal(shape=(self.batch_size,) + self.input_map.shape)

    #     # sample uniform random diffusion times
    #     diffusion_times = tf.random.uniform(
    #         shape=[self.batch_size, ] + [1] * self.input_map.axes(), minval=0.0, maxval=1.0
    #     )
    #     noise_rates, signal_rates = self.diffusion_schedule(diffusion_times)
    #     # mix the images with noises accordingly
    #     noisy_images = signal_rates * images + noise_rates * noises

    #     # use the network to separate noisy images to their components
    #     pred_noises, pred_images = self.denoise(
    #         control_embed, noisy_images, noise_rates, signal_rates, training=False
    #     )

    #     noise_loss = self.loss(noises, pred_noises)
    #     image_loss = self.loss(images, pred_images)

    #     self.image_loss_tracker.update_state(image_loss)
    #     self.noise_loss_tracker.update_state(noise_loss)

    #     # measure KID between real and generated images
    #     # this is computationally demanding, kid_diffusion_steps has to be small
    #     images = self.denormalize(images)
    #     generated_images = self.generate(
    #         control_embed, num_images=self.batch_size, diffusion_steps=20
    #     )
    #     return generated_images

    def test_step(self, batch):
        # normalize images to have standard deviation of 1, like the noises
        images = batch[0][self.input_map.input_name()]
        self.normalizer.update_state(images)
        images = self.normalizer(images, training=False)

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

        self.image_loss_tracker.update_state(image_loss)
        self.noise_loss_tracker.update_state(noise_loss)

        # measure KID between real and generated images
        # this is computationally demanding, kid_diffusion_steps has to be small
        images = self.denormalize(images)
        generated_images = self.generate(
            control_embed, num_images=self.batch_size, diffusion_steps=20,
        )
        #         self.kid.update_state(images, generated_images)

        return {m.name: m.result() for m in self.metrics}

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
        epoch=None, logs=None, num_rows=4, num_cols=4,
    ):
        images = batch[0][self.input_map.input_name()]
        self.normalizer.update_state(images)
        images = self.normalizer(images, training=False)
        noises = tf.random.normal(shape=(self.batch_size,) + self.input_map.shape)
        diffusion_times = diffusion_amount * tf.ones(shape=[self.batch_size] + [1] * self.input_map.axes())
        noise_rates, signal_rates = self.diffusion_schedule(diffusion_times)
        # mix the images with noises accordingly
        noisy_images = signal_rates * images + noise_rates * noises

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
                plt.imshow(generated_images[index], cmap='gray')
                plt.axis("off")
        plt.tight_layout()
        plt.show()
        plt.close()

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


def get_network(input_shape, widths, block_depth, kernel_size):
    noisy_images = keras.Input(shape=input_shape)
    conv, upsample, pool = layers_from_shape(input_shape)
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

        self.image_loss_tracker.update_state(image_loss)
        self.noise_loss_tracker.update_state(noise_loss)

        # measure KID between real and generated images
        # this is computationally demanding, kid_diffusion_steps has to be small
        images = self.denormalize(images)
        generated_images = self.generate(
            num_images=self.batch_size, diffusion_steps=20,
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

    def plot_ecgs(self, epoch=None, logs=None, num_rows=2, num_cols=8, reseed=None, prefix='./figures/'):
        # plot random generated images for visual evaluation of generation quality
        generated_images = self.generate(
            num_images=max(self.batch_size, num_rows * num_cols),
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
        now_string = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')
        figure_path = os.path.join(prefix, f'diffusion_ecg_generations_{now_string}{IMAGE_EXT}')
        if not os.path.exists(os.path.dirname(figure_path)):
            os.makedirs(os.path.dirname(figure_path))
        plt.savefig(figure_path, bbox_inches="tight")

    def plot_reconstructions(
        self, images_original, diffusion_amount=0,
        epoch=None, logs=None, num_rows=3, num_cols=6,
    ):
        images = images_original[0][self.tensor_map.input_name()]
        self.normalizer.update_state(images)
        images = self.normalizer(images, training=False)
        noises = tf.random.normal(shape=(self.batch_size,) + self.tensor_map.shape)

        diffusion_times = diffusion_amount * tf.ones(shape=[self.batch_size] + [1] * self.tensor_map.axes())
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
        plt.show()
        plt.close()

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
                learning_rate=learning_rate, weight_decay=weight_decay,
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
