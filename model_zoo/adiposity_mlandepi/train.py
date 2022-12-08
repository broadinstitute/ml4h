# ML4H is released under the following BSD 3-Clause License:
#
# Copyright (c) 2020, Broad Institute, Inc. and The General Hospital Corporation.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name Broad Institute, Inc. or The General Hospital Corporation
#   nor the names of its contributors may be used to endorse or promote products
#   derived from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
import os
import cv2
import time
import tensorflow as tf
import numpy as np
import h5py
import blosc
import ShrinkageLoss from shrinkage_loss
from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    TensorBoard,
)

# Helper functions
def rotate_image(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result  = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result


def translate_image(image, shape, steps):
    M = np.float32([[1, 0, steps], [0, 1, steps]])
    image = cv2.warpAffine(image, M, shape)
    return image


def uncompress_data(compressed_data: h5py.Dataset, stored_dtype = np.uint16) -> np.ndarray:
    return np.frombuffer(
        blosc.decompress(compressed_data[()]), dtype=stored_dtype
    ).reshape(compressed_data.attrs["shape"]).astype(np.float32)


def Top50Normalize(tensor: np.ndarray) -> np.ndarray:
        upper = np.mean(sorted(np.max(tensor, axis=-1).flatten())[::-1][0:50])
        tensor = np.where(tensor >= upper, upper, tensor)
        tensor /= tensor.max()
        return tensor


# Dataloader callback function for retrieving and processing the projections for training
def mdrk_projection_both_views_pretrained(
    instance: int = 2,
    augment: bool = False,
    stationwise_normalization=True,
    normalize_histogram=True,
    clahe_amount=5,
    clahe_clip=2.0,
):
    """This function wrapper constructs a new image with the coronal and sagittal 2D
    projections side-by-side for the water/fat reconstructions stacked in the
    channels followed by an empty channel to fit the expectations of pretrained image
    models. Returns a (237, 256, 3) tensor.

    Requirements:

    This subroutine requires that the target HDF5 file has the following datasets:
    * /instance/{instance}/w_sagittal and /instance/{instance}/w_coronal
    * /instance/{instance}/f_sagittal and /instance/{instance}/f_coronal

    that are compressed with blosc as 16-bit unsigned integers. Each dataset must also
    have the attribute `shape`.

    Args:
        instance (int, optional): UK Biobank instance numbering. Defaults to 2.
        augment (bool, optional): Augment data: includes a translation, rotation, and axis flip. Defaults to False.
        stationwise_normalization (bool, optional): Normalize each station separately before appending as a channel. Defaults to True.
        normalize_histogram (bool, optional): Normalize intensity histogram using CLAHE. Defaults to True.
        clahe_amount (int, optional): Size of CLAHE kernel. Defaults to 5.
        clahe_clip (float, optional): Clip limit for the CLAHE kernel. Defaults to 2.0.
    """
    def _mdrk_projection_both_views_pretrained(tm, hd5, dependents={}):
        #
        do_augment = False
        do_flip = False
        rand_angle = 0.0
        rand_move = 0.0
        cclip = clahe_clip
        camount = clahe_amount
        if augment:
            do_augment = True
            if np.random.random() > 0.5:
                do_flip = True
            rand_angle = np.random.randint(-5, 5)
            rand_move = np.random.randint(-16, 16)
            cclip = np.random.randint(0, 5)
            camount = np.random.randint(1, 10)
        #
        clahe = cv2.createCLAHE(
            clipLimit=cclip, tileGridSize=(camount, camount)
        )
        prefixes = ["w", "f"]
        tensor = np.zeros((368, 174 + 224, 3), dtype=np.float32)
        #
        for p, i in zip(prefixes, range(len(prefixes))):
            # Coronal view
            try:
                compressed_data = hd5["instance"][str(instance)][f"{p}_coronal"]
            except Exception as e:
                raise Exception(e)
            #
            try:
                tensor_coronal = uncompress_data(compressed_data)
            except Exception as e:
                raise Exception(e)
            #
            if stationwise_normalization:
                tensor_coronal = Top50Normalize(tensor_coronal)
            if normalize_histogram:
                tensor_coronal = (
                    clahe.apply((tensor_coronal * 255.0).astype(np.uint8)).astype(
                        np.float32
                    )
                    / 255.0
                )
            #
            tensor_coronal = cv2.resize(tensor_coronal, (224, 368))
            if do_augment:
                if do_flip:
                    tensor_coronal = cv2.flip(tensor_coronal, 1)
                tensor_coronal = translate_image(tensor_coronal, (224, 368), rand_move)
                tensor_coronal = rotate_image(tensor_coronal, rand_angle)
            tensor[..., 0:224, i] = tensor_coronal
            #
            try:
                compressed_data = hd5["instance"][str(instance)][f"{p}_sagittal"]
            except Exception as e:
                raise Exception(e)
            # Sagittal view
            try:
                tensor_sagittal = uncompress_data(compressed_data)
            except Exception as e:
                raise Exception(e)
            #
            if stationwise_normalization:
                tensor_sagittal = Top50Normalize(tensor_sagittal)
            if normalize_histogram:
                tensor_sagittal = (
                    clahe.apply((tensor_sagittal * 255.0).astype(np.uint8)).astype(
                        np.float32
                    )
                    / 255.0
                )
            #
            tensor_sagittal = cv2.resize(tensor_sagittal, (174, 368))
            if do_augment:
                tensor_sagittal = translate_image(
                    tensor_sagittal, (174, 368), rand_move
                )
                tensor_sagittal = rotate_image(tensor_sagittal, rand_angle)
            tensor[..., 224:, i] = tensor_sagittal
        tensor = cv2.resize(tensor, (256, 237))
        return tensor
    return _mdrk_projection_both_views_pretrained


# Load your train and test data splits here
train_data = ...
test_data = ...

# Training and testing datasets using `mdrk_projection_both_views_pretrained` callback and the
# target phenotype (ASAT, VAT, GFAT, TAT) as the label
train_ds = ...
test_ds = ...

# Metrics we track during training
METRICS = [
    tf.keras.metrics.MeanSquaredError(name="mean_squared_error", dtype=None),
    tf.keras.metrics.RootMeanSquaredError(name="root_mean_squared_error", dtype=None),
    tf.keras.metrics.MeanAbsoluteError(name="mean_absolute_error", dtype=None),
    tf.keras.metrics.MeanSquaredLogarithmicError(
        name="mean_squared_logarithmic_error", dtype=None
    ),
    tf.keras.metrics.LogCoshError(name="logcosh", dtype=None),
]

# Model
input = tf.keras.Input(
    shape=(237, 256, 3),
    name="mri_projection",
)

base_model = tf.keras.applications.densenet.DenseNet121(
    include_top=False,
    weights='imagenet',
    input_tensor=input,
    pooling='max',
    classes=1,
)
# Downloading data from https://storage.googleapis.com/tensorflow/keras-applications/densenet/densenet121_weights_tf_dim_ordering_tf_kernels_notop.h5
base_model.trainable = True  # Freeze base model if set to `False`

# Replace final layer with MLP layers
x = tf.keras.layers.Dense(128)(base_model.layers[-1].output)
x = tf.keras.layers.Activation("relu")(x)
x = tf.keras.layers.Dense(512)(x)
x = tf.keras.layers.Activation("relu")(x)

# Output regression layer
output = tf.keras.layers.Dense(
    1, name="output_mdrk_adiposity_scalar_output_fake_continuous"
)(x)

model = tf.keras.models.Model(inputs=input, outputs=output)

########

loss = ShrinkageLoss(a=5.0, c=0.2)
opt = tf.keras.optimizers.Adam(beta_1=0.9, beta_2=0.999, epsilon=1e-08, clipnorm=1.0)
model.compile(optimizer=opt, loss=loss, metrics=METRICS)

# Output path and model path
output_path = ...
model_file = ...

n_epochs = 100
decay = tf.keras.experimental.CosineDecay(
    initial_learning_rate=1e-4, decay_steps=n_epochs, alpha=0.0
)

lrate = tf.keras.callbacks.LearningRateScheduler(decay, verbose=1)

callback_list = [
    lrate,
    ModelCheckpoint(
        filepath=model_file,
        monitor="val_mean_absolute_error",
        verbose=1,
        save_best_only=True,
    ),
    TensorBoard(
        log_dir=output_path,
        histogram_freq=1,
        write_graph=True,
        write_images=True,
        update_freq=10,
        profile_batch=0,
    ),
    # EarlyStopping(monitor="val_mean_absolute_error", patience=30, verbose=1, mode="auto"),
]

history = model.fit(
    train_ds,
    validation_data=test_ds,
    epochs=n_epochs,
    verbose=1,
    callbacks=callback_list,
)
