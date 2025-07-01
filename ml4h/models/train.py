# Training models built by the model factory

import os
import logging
from functools import partial
from typing import List, Tuple, Iterable, Union
from google.cloud import storage

import tensorflow as tf
import keras


import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

import tensorflow as tf
from ml4h.explorations import predictions_to_pngs

from tensorflow.keras.callbacks import History
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, Callback, CSVLogger, TensorBoard

from ml4h.TensorMap import TensorMap

from ml4h.models.diffusion_blocks import DiffusionModel, DiffusionController
from ml4h.plots import plot_metric_history, plot_roc
from ml4h.defines import IMAGE_EXT, MODEL_EXT
from ml4h.models.inspect import plot_and_time_model
from ml4h.models.model_factory import get_custom_objects, make_multimodal_multitask_model
from ml4h.tensor_generators import test_train_valid_tensor_generators, big_batch_from_minibatch_generator


def train_model_from_generators(
    model: Model,
    generate_train: Iterable,
    generate_valid: Iterable,
    training_steps: int,
    validation_steps: int,
    batch_size: int,
    epochs: int,
    patience: int,
    output_folder: str,
    run_id: str,
    inspect_model: bool,
    inspect_show_labels: bool,
    output_tensor_maps: List[TensorMap] = [],
    return_history: bool = False,
    plot: bool = True,
    save_last_model: bool = False,
    gcp_bucket: str = None,
    gcp_path: str = None,
    log_tensorboard: bool = False,
) -> Union[Model, Tuple[Model, History]]:
    """Train a model from tensor generators for validation and training data.

    Training data lives on disk, it will be loaded by generator functions.
    Plots the metric history after training. Creates a directory to save weights, if necessary.
    Measures runtime and plots architecture diagram if inspect_model is True.
    :param model: The model to optimize
    :param generate_train: Generator function that yields mini-batches of training data.
    :param generate_valid: Generator function that yields mini-batches of validation data.
    :param training_steps: Number of mini-batches in each so-called epoch
    :param validation_steps: Number of validation mini-batches to examine after each epoch.
    :param batch_size: Number of training examples in each mini-batch
    :param epochs: Maximum number of epochs to run regardless of Early Stopping
    :param patience: Number of epochs to wait before reducing learning rate.
    :param output_folder: Directory where output file will be stored
    :param run_id: User-chosen string identifying this run
    :param inspect_model: If True, measure training and inference runtime of the model and generate architecture plot.
    :param inspect_show_labels: If True, show labels on the architecture plot.
    :param output_tensor_maps: List of output TensorMap
    :param return_history: If true return history from training and don't plot the training history
    :param plot: If true, plots the metrics for train and validation set at the end of each epoch
    :param save_last_model: If true saves the model weights from last epoch otherwise saves model with best validation loss
    :param gcp_bucket: If not None, save the model to GCP bucket with gcp_path (should be without gs://)
    :param gcp_path: If gcp_bucket is not None, save the model to gs://gcp_bucket/gcp_path
    :param log_tensorboard: If True, create TensorBorad logging

    :return: The optimized model.

    """
    model_file = os.path.join(output_folder, run_id, run_id + MODEL_EXT)
    if not os.path.exists(os.path.dirname(model_file)):
        os.makedirs(os.path.dirname(model_file))

    if inspect_model:
        image_p = os.path.join(output_folder, run_id, 'architecture_graph_' + run_id + IMAGE_EXT)
        plot_and_time_model(model, generate_train, generate_valid, batch_size, training_steps, inspect_show_labels, image_p)

    history = model.fit(
        generate_train, steps_per_epoch=training_steps, epochs=epochs, verbose=1,
        validation_steps=validation_steps, validation_data=generate_valid,
        callbacks=_get_callbacks(patience, model_file, save_last_model, log_tensorboard, gcp_bucket, gcp_path),
    )

    logging.info('Model weights saved at: %s' % model_file)
    custom_dict = get_custom_objects(output_tensor_maps)
    model = tf.keras.models.load_model(model_file, custom_objects=custom_dict, compile=False)
    model.compile(optimizer='adam', loss='mse')
    if plot:
        plot_metric_history(history, training_steps, run_id, os.path.dirname(model_file))
    if return_history:
        return model, history
    return model


def _get_callbacks(
    patience: int, model_file: str, save_last_model: bool,
) -> List[Callback]:
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=patience * 3, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=patience, verbose=1),
        ModelCheckpoint(filepath=model_file, verbose=1, save_best_only=not save_last_model),
        ModelCheckpointWithGCP(model_file, gcp_bucket, gcp_path, verbose=1, save_best_only=not save_last_model) if gcp_bucket else None,
        CSVLogger(os.path.join(os.path.dirname(model_file), 'training_log.csv'), append=True),
        TensorBoard(log_dir=os.path.join(os.path.dirname(model_file), 'tensorboard_logs'),
                    histogram_freq=1) if log_tensorboard else None,
    ]
    callbacks = [item for item in callbacks if item is not None]

    return callbacks


### Custom Callbacks - for logging training and copying to GCP bucket midrun ###
class ModelCheckpointWithGCP(Callback):
    def __init__(self, local_checkpoint_path, gcp_bucket, gcp_path, monitor='val_loss', verbose=0, save_best_only=False, mode='auto'):
        super().__init__()
        self.local_checkpoint_path = local_checkpoint_path
        self.gcp_bucket = gcp_bucket
        self.gcp_path = gcp_path
        self.save_best_only = save_best_only
        self.monitor = monitor
        self.mode = mode

        # Google Cloud Storage client
        self.client = storage.Client()
        self.bucket = self.client.bucket(self.gcp_bucket)

        # For tracking best value
        self.best = None
        if self.mode == 'min':
            self.monitor_op = np.less
            self.best = np.inf
        elif self.mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.inf
        else:
            if 'acc' in self.monitor or self.monitor.startswith('fmeasure'):
                self.monitor_op = np.greater
                self.best = -np.inf
            else:
                self.monitor_op = np.less
                self.best = np.inf

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        current = logs.get(self.monitor)
        if current is None:
            return

        # Upload the file to GCP if it's the best or not save_best_only
        if (not self.save_best_only) or self.monitor_op(current, self.best):
            self.best = current
            self.upload_to_gcp()

    def upload_to_gcp(self):
        try:
            blob = self.bucket.blob(os.path.join(self.gcp_path,os.path.basename(self.local_checkpoint_path)))
            blob.upload_from_filename(self.local_checkpoint_path)
            print(f"Uploaded checkpoint to gs://{self.gcp_bucket}/{self.gcp_path}")

            if os.path.exists(os.path.join(os.path.dirname(self.local_checkpoint_path), 'training_log.csv')):
                blob_log = self.bucket.blob(os.path.join(self.gcp_path,'training_log.csv'))
                blob_log.upload_from_filename(os.path.join(os.path.dirname(self.local_checkpoint_path),'training_log.csv'))
                print(f"Uploaded training log to gs://{self.gcp_bucket}/{self.gcp_path}/training_log.csv")
        except Exception as e:
            print(f"Failed to upload checkpoint: {e}")
