# Training models built by the model factory

import os
import logging
from typing import List, Tuple, Iterable, Union

import tensorflow as tf
from tensorflow import keras
import tensorflow_addons as tfa
from tensorflow.keras.callbacks import History
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, Callback

from ml4h.TensorMap import TensorMap
from ml4h.models.diffusion_blocks import DiffusionModel
from ml4h.plots import plot_metric_history
from ml4h.defines import IMAGE_EXT, MODEL_EXT
from ml4h.models.inspect import plot_and_time_model
from ml4h.models.model_factory import get_custom_objects
from ml4h.tensor_generators import test_train_valid_tensor_generators


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
        callbacks=_get_callbacks(patience, model_file, save_last_model),
    )

    logging.info('Model weights saved at: %s' % model_file)
    custom_dict = get_custom_objects(output_tensor_maps)
    model = load_model(model_file, custom_objects=custom_dict, compile=False)
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
    ]
    
    return callbacks


def train_diffusion_model(args):
    generate_train, generate_valid, generate_test = test_train_valid_tensor_generators(**args.__dict__)
    diffusion_model = DiffusionModel(args.tensor_maps_in[0], args.dense_blocks, args.block_size, args.conv_x)

    diffusion_model.compile(
        optimizer=tfa.optimizers.AdamW(
            learning_rate=args.learning_rate, weight_decay=1e-4
        ),
        loss=keras.losses.mean_absolute_error,
    )
    batch = next(generate_train)
    for k in batch[0]:
        logging.info(f"input {k} {batch[0][k].shape}")
        feature_batch = batch[0][k]
    for k in batch[1]:
        logging.info(f"label {k} {batch[1][k].shape}")
    checkpoint_path = f"{args.output_folder}/{args.id}/checkpoints"
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        save_weights_only=True,
        monitor="val_i_loss",
        mode="min",
        save_best_only=True,
    )

    # calculate mean and variance of training dataset for normalization
    diffusion_model.normalizer.adapt(feature_batch)
    if args.inspect_model:
        tf.keras.utils.plot_model(
            diffusion_model.network,
            to_file=f"{args.output_folder}/{args.id}/architecture_diffusion_unet.png",
            show_shapes=True,
            show_dtype=False,
            show_layer_names=True,
            rankdir="TB",
            expand_nested=True,
            dpi=args.dpi,
            layer_range=None,
            show_layer_activations=False,
        )
    history = diffusion_model.fit(
        generate_train,
        steps_per_epoch=args.training_steps,
        epochs=args.epochs,
        validation_data=generate_valid,
        validation_steps=args.validation_steps,
        callbacks=[
            #keras.callbacks.LambdaCallback(on_epoch_end=diffusion_model.plot_images),
            checkpoint_callback,
        ],
    )
    diffusion_model.load_weights(checkpoint_path)
    #diffusion_model.compile(optimizer='adam', loss='mse')
    plot_metric_history(history, args.training_steps, args.id, os.path.dirname(checkpoint_path))
    if args.inspect_model:
        diffusion_model.plot_images(num_rows=4, prefix=os.path.dirname(checkpoint_path)+'/final_')
    return diffusion_model
