# models.py

# Imports
import os
import h5py
import time
import logging
import operator
import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple, Iterable, Callable, Union

# Keras imports
import keras.backend as K
from keras.callbacks import History
from keras.optimizers import Adam
from keras.models import Model, load_model
from keras.utils.vis_utils import model_to_dot
from keras.layers import LeakyReLU, PReLU, ELU, ThresholdedReLU, Lambda
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.layers import SpatialDropout1D, SpatialDropout2D, SpatialDropout3D, add, concatenate
from keras.layers import Input, Dense, Dropout, BatchNormalization, Activation, Flatten, LSTM, RepeatVector
from keras.layers import Conv1D, Conv2D, Conv3D, UpSampling1D, UpSampling2D, UpSampling3D, MaxPooling1D
from keras.layers import MaxPooling2D, MaxPooling3D, AveragePooling1D, AveragePooling2D, AveragePooling3D, Layer
from keras.layers import SeparableConv1D, SeparableConv2D, DepthwiseConv2D

from ml4cvd.TensorMap import TensorMap
from ml4cvd.metrics import get_metric_dict
from ml4cvd.plots import plot_metric_history
from ml4cvd.defines import JOIN_CHAR, IMAGE_EXT, TENSOR_EXT, ECG_CHAR_2_IDX
from ml4cvd.optimizers import get_optimizer
from ml4cvd.lookahead import Lookahead

CHANNEL_AXIS = -1  # Set to 1 for Theano backend


def make_shallow_model(tensor_maps_in: List[TensorMap], tensor_maps_out: List[TensorMap],
                       learning_rate: float, model_file: str = None, model_layers: str = None) -> Model:
    """Make a shallow model (e.g. linear or logistic regression)

	Input and output tensor maps are set from the command line.
	Model summary printed to output

    :param tensor_maps_in: List of input TensorMaps, only 1 input TensorMap is currently supported,
                            otherwise there are layer name collisions.
    :param tensor_maps_out: List of output TensorMaps
    :param learning_rate: Size of learning steps in SGD optimization
    :param model_file: Optional HD5 model file to load and return.
    :param model_layers: Optional HD5 model file whose weights will be loaded into this model when layer names match.
    :return: a compiled keras model
    """
    if model_file is not None:
        m = load_model(model_file, custom_objects=get_metric_dict(tensor_maps_out))
        m.summary()
        logging.info("Loaded model file from: {}".format(model_file))
        return m

    losses = []
    outputs = []
    my_metrics = {}
    loss_weights = []
    input_tensors = [Input(shape=tm.shape, name=tm.input_name()) for tm in tensor_maps_in]
    if len(input_tensors) > 1:
        logging.warning('multi input tensors not fully supported')
    for it in input_tensors:
        for ot in tensor_maps_out:
            losses.append(ot.loss)
            loss_weights.append(ot.loss_weight)
            my_metrics[ot.output_name()] = ot.metrics
            outputs.append(Dense(units=len(ot.channel_map), activation=ot.activation, name=ot.output_name())(it))

    opt = Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    m = Model(inputs=input_tensors, outputs=outputs)
    m.compile(optimizer=opt, loss=losses, loss_weights=loss_weights, metrics=my_metrics)
    m.summary()

    if model_layers is not None:
        m.load_weights(model_layers, by_name=True)
        logging.info('Loaded model weights from:{}'.format(model_layers))

    return m


def make_waveform_model_unet(tensor_maps_in: List[TensorMap], tensor_maps_out: List[TensorMap], learning_rate: float,
                             model_file: str = None, model_layers: str = None) -> Model:
    """Make a waveform predicting model

	Input and output tensor maps are set from the command line.
	Model summary printed to output

    :param tensor_maps_in: List of input TensorMaps, only 1 input TensorMap is currently supported,
                            otherwise there are layer name collisions.
    :param tensor_maps_out: List of output TensorMaps
    :param learning_rate: Size of learning steps in SGD optimization
    :param model_file: Optional HD5 model file to load and return.
    :param model_layers: Optional HD5 model file whose weights will be loaded into this model when layer names match.
    :return: a compiled keras model
    """
    if model_file is not None:
        m = load_model(model_file, custom_objects=get_metric_dict(tensor_maps_out))
        m.summary()
        logging.info("Loaded model file from: {}".format(model_file))
        return m

    neurons = 24
    input_tensor = residual = Input(shape=tensor_maps_in[0].shape, name=tensor_maps_in[0].input_name())
    x = c600 = Conv1D(filters=neurons, kernel_size=11, activation='relu', padding='same')(input_tensor)
    x = Conv1D(filters=neurons, kernel_size=51, activation='relu', padding='same')(x)
    x = MaxPooling1D(2)(x)
    x = c300 = Conv1D(filters=neurons, kernel_size=111, activation='relu', padding='same')(x)
    x = Conv1D(filters=neurons, kernel_size=201, activation='relu', padding='same')(x)
    x = MaxPooling1D(2)(x)
    x = Conv1D(filters=neurons, kernel_size=301, activation='relu', padding='same')(x)
    x = Conv1D(filters=neurons, kernel_size=301, activation='relu', padding='same')(x)
    x = UpSampling1D(2)(x)
    x = concatenate([x, c300])
    x = Conv1D(filters=neurons, kernel_size=201, activation='relu', padding='same')(x)
    x = UpSampling1D(2)(x)
    x = Conv1D(filters=neurons, kernel_size=111, activation='relu', padding='same')(x)
    x = concatenate([x, c600])
    x = Conv1D(filters=neurons, kernel_size=51, activation='relu', padding='same')(x)
    x = concatenate([x, residual])
    conv_label = Conv1D(filters=tensor_maps_out[0].shape[CHANNEL_AXIS], kernel_size=1, activation="linear")(x)
    output_y = Activation(tensor_maps_out[0].activation, name=tensor_maps_out[0].output_name())(conv_label)
    m = Model(inputs=[input_tensor], outputs=[output_y])
    m.summary()
    opt = Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, clipnorm=1.0)
    m.compile(optimizer=opt, loss=tensor_maps_out[0].loss, metrics=tensor_maps_out[0].metrics)

    if model_layers is not None:
        m.load_weights(model_layers, by_name=True)
        logging.info('Loaded model weights from:{}'.format(model_layers))

    return m


def make_character_model_plus(tensor_maps_in: List[TensorMap], tensor_maps_out: List[TensorMap], learning_rate: float,
                              base_model: Model, model_layers: str=None) -> Tuple[Model, Model]:
    """Make a ECG captioning model from an ECG embedding model

    The base_model must have an embedding layer, but besides that can have any number of other predicition TensorMaps.
    Input and output tensor maps are set from the command line.
    Model summary printed to output

    :param tensor_maps_in: List of input TensorMaps, only 1 input TensorMap is currently supported,
                            otherwise there are layer name collisions.
    :param tensor_maps_out: List of output TensorMaps
    :param learning_rate: Size of learning steps in SGD optimization
    :param base_model: The model the computes the ECG embedding
    :param model_layers: Optional HD5 model file whose weights will be loaded into this model when layer names match.
    :return: a tuple of the compiled keras model and the character emitting sub-model
    """
    char_maps_in, char_maps_out = _get_tensor_maps_for_characters(tensor_maps_in, base_model)
    tensor_maps_in.extend(char_maps_in)
    tensor_maps_out.extend(char_maps_out)
    char_model = make_character_model(tensor_maps_in, tensor_maps_out, learning_rate)
    losses = []
    my_metrics = {}
    loss_weights = []
    output_layers = []
    for tm in tensor_maps_out:
        losses.append(tm.loss)
        loss_weights.append(tm.loss_weight)
        my_metrics[tm.output_name()] = tm.metrics
        if tm.name == 'ecg_rest_next_char':
            output_layers.append(char_model.get_layer(tm.output_name()))
        else:
            output_layers.append(base_model.get_layer(tm.output_name()))

    m = Model(inputs=base_model.inputs+char_model.inputs, outputs=base_model.outputs+char_model.outputs)
    m.summary()
    opt = Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, clipnorm=1.0)
    m.compile(optimizer=opt, loss=losses, loss_weights=loss_weights, metrics=my_metrics)

    if model_layers is not None:
        m.load_weights(model_layers, by_name=True)
        logging.info('Loaded model weights from:{}'.format(model_layers))

    return m, char_model


def make_character_model(tensor_maps_in: List[TensorMap], tensor_maps_out: List[TensorMap], learning_rate: float,
                         model_file: str = None, model_layers: str = None) -> Model:
    """Make a ECG captioning model

	Input and output tensor maps are set from the command line.
	Model summary printed to output

    :param tensor_maps_in: List of input TensorMaps, only 1 input TensorMap is currently supported,
                            otherwise there are layer name collisions.
    :param tensor_maps_out: List of output TensorMaps
    :param learning_rate: Size of learning steps in SGD optimization
    :param model_file: Optional HD5 model file to load and return.
    :param model_layers: Optional HD5 model file whose weights will be loaded into this model when layer names match.
    :return: a compiled keras model
    """
    if model_file is not None:
        m = load_model(model_file, custom_objects=get_metric_dict(tensor_maps_out))
        m.summary()
        logging.info("Loaded model file from: {}".format(model_file))
        return m

    input_layers = []
    for it in tensor_maps_in:
        if it.is_hidden_layer():
            embed_in = Input(shape=it.shape, name=it.input_name())
            input_layers.append(embed_in)
        elif it.is_ecg_text():
            burn_in = Input(shape=it.shape, name=it.input_name())
            input_layers.append(burn_in)
            repeater = RepeatVector(it.shape[0])
        else:
            logging.warning(f"character model cant handle {it.name} from group:{it.group}")

    logging.info(f"inputs: {[il.name for il in input_layers]}")
    wave_embeds = repeater(embed_in)
    lstm_in = concatenate([burn_in, wave_embeds], name='concat_embed_and_text')
    lstm_out = LSTM(128)(lstm_in)

    output_layers = []
    for ot in tensor_maps_out:
        if ot.name == 'ecg_rest_next_char':
            output_layers.append(Dense(ot.shape[-1], activation=ot.activation, name=ot.output_name())(lstm_out))

    m = Model(inputs=input_layers, outputs=output_layers)
    m.summary()
    opt = Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, clipnorm=1.0)
    m.compile(optimizer=opt, loss='categorical_crossentropy')

    if model_layers is not None:
        m.load_weights(model_layers, by_name=True)
        logging.info('Loaded model weights from:{}'.format(model_layers))

    return m


def make_siamese_model(base_model: Model,
                       tensor_maps_in: List[TensorMap],
                       hidden_layer: str,
                       learning_rate: float = None,
                       optimizer: str = 'adam',
                       **kwargs) -> Model:
    in_left = [Input(shape=tm.shape, name=tm.input_name()+'_left') for tm in tensor_maps_in]
    in_right = [Input(shape=tm.shape, name=tm.input_name()+'_right') for tm in tensor_maps_in]
    encode_model = make_hidden_layer_model(base_model, tensor_maps_in, hidden_layer)
    h_left = encode_model(in_left)
    h_right = encode_model(in_right)

    # Compute the L1 distance
    l1_layer = Lambda(lambda tensors: K.abs(tensors[0] - tensors[1]))
    l1_distance = l1_layer([h_left, h_right])

    # Add a dense layer with a sigmoid unit to generate the similarity score
    prediction = Dense(1, activation='sigmoid', name='output_siamese')(l1_distance)

    m = Model(inputs=in_left+in_right, outputs=prediction)
    opt = get_optimizer(optimizer, learning_rate, kwargs.get('optimizer_kwargs'))
    m.compile(optimizer=opt, loss='binary_crossentropy')

    if kwargs['model_layers'] is not None:
        m.load_weights(kwargs['model_layers'], by_name=True)
        logging.info(f"Loaded model weights from:{kwargs['model_layers']}")

    return m


def make_hidden_layer_model_from_file(parent_file: str, tensor_maps_in: List[TensorMap], output_layer_name: str, tensor_maps_out: List[TensorMap]) -> Model:
    parent_model = load_model(parent_file, custom_objects=get_metric_dict(tensor_maps_out))
    return make_hidden_layer_model(parent_model, tensor_maps_in, output_layer_name)


def make_hidden_layer_model(parent_model: Model, tensor_maps_in: List[TensorMap], output_layer_name: str) -> Model:
    parent_inputs = [parent_model.get_layer(tm.input_name()).input for tm in tensor_maps_in]
    dummy_input = {tm.input_name(): np.zeros((1,) + parent_model.get_layer(tm.input_name()).input_shape[1:]) for tm in tensor_maps_in}
    intermediate_layer_model = Model(inputs=parent_inputs, outputs=parent_model.get_layer(output_layer_name).output)
    # If we do not predict here then the graph is disconnected, I do not know why?!
    intermediate_layer_model.predict(dummy_input)
    return intermediate_layer_model


def make_multimodal_multitask_model(tensor_maps_in: List[TensorMap]=None,
                                    tensor_maps_out: List[TensorMap]=None,
                                    activation: str=None,
                                    dense_layers: List[int]=None,
                                    dropout: float=None,
                                    mlp_concat: bool=None,
                                    conv_layers: List[int]=None,
                                    max_pools: List[int]=None,
                                    res_layers: List[int]=None,
                                    dense_blocks: List[int]=None,
                                    block_size: List[int]=None,
                                    conv_type: str=None,
                                    conv_normalize: str=None,
                                    conv_regularize: str=None,
                                    conv_x: int=None,
                                    conv_y: int=None,
                                    conv_z: int=None,
                                    conv_dropout: float=None,
                                    conv_width: int=None,
                                    conv_dilate: bool=None,
                                    u_connect: bool=None,
                                    pool_x: int=None,
                                    pool_y: int=None,
                                    pool_z: int=None,
                                    pool_type: int=None,
                                    padding: str=None,
                                    learning_rate: float=None,
                                    optimizer: str='adam',
                                    **kwargs) -> Model:
    """Make multi-task, multi-modal feed forward neural network for all kinds of prediction

	This model factory can be used to make networks for classification, regression, and segmentation
	The tasks attempted are given by the output TensorMaps.
	The modalities and the first layers in the architecture are determined by the input TensorMaps.

	Hyperparameters are exposed to the command line.
	Model summary printed to output

    :param model_file: HD5 model file to load and return.
    :param model_layers: HD5 model file whose weights will be loaded into this model when layer names match.
    :param freeze_model_layers: Whether to freeze layers from loaded from model_layers
    :param tensor_maps_in: List of input TensorMaps
    :param tensor_maps_out: List of output TensorMaps
    :param activation: Activation function as a string (e.g. 'relu', 'linear, or 'softmax)
    :param dense_layers: List of number of filters in each dense layer.
    :param dropout: Dropout rate in dense layers
    :param mlp_concat: If True, conncatenate unstructured inputs to each deeper dense layer
    :param conv_layers: List of number of filters in each convolutional layer
    :param max_pools: List of maxpool sizes in X and Y dimensions after convolutional layers
    :param res_layers: List of convolutional layers with residual connections
    :param dense_blocks: List of number of filters in densenet modules for densenet convolutional models
    :param block_size: Number of layers within each Densenet module for densenet convolutional models
    :param conv_bn: if True, Batch normalize convolutional layers
    :param conv_x: Size of X dimension for 2D and 3D convolutional kernels
    :param conv_y: Size of Y dimension for 2D and 3D convolutional kernels
    :param conv_z: Size of Z dimension for 3D convolutional kernels
    :param conv_dropout: Dropout rate in convolutional layers
    :param conv_width: Size of convolutional kernel for 1D models.
    :param u_connect: Include U connections between early and late convolutional layers.
    :param pool_x: Pooling in the X dimension for Convolutional models.
    :param pool_y: Pooling in the Y dimension for Convolutional models.
    :param pool_z: Pooling in the Z dimension for 3D Convolutional models.
    :param padding: Padding string can be 'valid' or 'same'. UNets and residual nets require 'same'.
    :param learning_rate:
    :param optimizer: which optimizer to use. See optimizers.py.
    :return: a compiled keras model
    """
    opt = get_optimizer(optimizer, learning_rate, kwargs.get('optimizer_kwargs'))
    metric_dict = get_metric_dict(tensor_maps_out)
    custom_dict = {**metric_dict, type(opt).__name__: opt}
    if 'model_file' in kwargs and kwargs['model_file'] is not None:
        logging.info("Attempting to load model file from: {}".format(kwargs['model_file']))
        m = load_model(kwargs['model_file'], custom_objects=custom_dict)
        m.summary()
        logging.info("Loaded model file from: {}".format(kwargs['model_file']))
        return m

    input_tensors = [Input(shape=tm.shape, name=tm.input_name()) for tm in tensor_maps_in]
    input_multimodal = []
    channel_axis = -1
    layers = {}

    for j, tm in enumerate(tensor_maps_in):
        if len(tm.shape) > 1:
            conv_fxns = _conv_layers_from_kind_and_dimension(len(tm.shape), conv_type, conv_layers, conv_width, conv_x, conv_y, conv_z, padding, conv_dilate)
            pool_layers = _pool_layers_from_kind_and_dimension(len(tm.shape), pool_type, len(max_pools), pool_x, pool_y, pool_z)
            last_conv = _conv_block_new(input_tensors[j], layers, conv_fxns, pool_layers, len(tm.shape), activation, conv_normalize, conv_regularize, conv_dropout, None)
            dense_conv_fxns = _conv_layers_from_kind_and_dimension(len(tm.shape), conv_type, dense_blocks, conv_width, conv_x, conv_y, conv_z, padding, False, block_size)
            dense_pool_layers = _pool_layers_from_kind_and_dimension(len(tm.shape), pool_type, len(dense_blocks), pool_x, pool_y, pool_z)
            last_conv = _dense_block(last_conv, layers, block_size, dense_conv_fxns, dense_pool_layers, len(tm.shape), activation, conv_normalize, conv_regularize, conv_dropout)
            input_multimodal.append(Flatten()(last_conv))
        else:
            mlp_input = input_tensors[j]
            mlp = _dense_layer(mlp_input, layers, tm.annotation_units, activation, conv_normalize)
            input_multimodal.append(mlp)

    if len(input_multimodal) > 1:
        multimodal_activation = concatenate(input_multimodal, axis=channel_axis)
    elif len(input_multimodal) == 1:
        multimodal_activation = input_multimodal[0]
    else:
        raise ValueError('No input activations.')

    for i, hidden_units in enumerate(dense_layers):
        if i == len(dense_layers) - 1:
            multimodal_activation = _dense_layer(multimodal_activation, layers, hidden_units, activation, conv_normalize, name='embed')
        else:
            multimodal_activation = _dense_layer(multimodal_activation, layers, hidden_units, activation, conv_normalize)
        if dropout > 0:
            multimodal_activation = Dropout(dropout)(multimodal_activation)
        if mlp_concat:
            multimodal_activation = concatenate([multimodal_activation, mlp_input], axis=channel_axis)

    losses = []
    my_metrics = {}
    loss_weights = []
    output_predictions = {}
    output_tensor_maps_to_process = tensor_maps_out.copy()

    while len(output_tensor_maps_to_process) > 0:
        tm = output_tensor_maps_to_process.pop(0)

        if not tm.parents is None and any(not p in output_predictions for p in tm.parents):
            output_tensor_maps_to_process.append(tm)
            continue

        losses.append(tm.loss)
        loss_weights.append(tm.loss_weight)
        my_metrics[tm.output_name()] = tm.metrics

        if len(tm.shape) > 1:
            all_filters = conv_layers + dense_blocks
            conv_layer, kernel = _conv_layer_from_kind_and_dimension(len(tm.shape), conv_type, conv_width, conv_x, conv_y, conv_z)
            for i, name in enumerate(reversed(_get_layer_kind_sorted(layers, 'Pooling'))):
                early_conv = _get_last_layer_by_kind(layers, 'Conv', int(name.split(JOIN_CHAR)[-1]))
                if u_connect:
                    last_conv = _upsampler(len(tm.shape), pool_x, pool_y, pool_z)(last_conv)
                    last_conv = conv_layer(filters=all_filters[-(1+i)], kernel_size=kernel, padding=padding)(last_conv)
                    last_conv = _activation_layer(activation)(last_conv)
                    last_conv = concatenate([last_conv, early_conv])
                else:
                    last_conv = _upsampler(len(tm.shape), pool_x, pool_y, pool_z)(last_conv)
                    last_conv = conv_layer(filters=all_filters[-(1 + i)], kernel_size=kernel, padding=padding)(last_conv)

            conv_label = conv_layer(tm.shape[channel_axis], _one_by_n_kernel(len(tm.shape)), activation="linear")(last_conv)
            output_predictions[tm.output_name()] = Activation(tm.activation, name=tm.output_name())(conv_label)
        elif tm.parents is not None:
            if len(K.int_shape(output_predictions[tm.parents[0]])) > 1:
                output_predictions[tm.output_name()] = Dense(units=tm.shape[0], activation=tm.activation, name=tm.output_name())(multimodal_activation)
            else:
                parented_activation = concatenate([multimodal_activation] + [output_predictions[p] for p in tm.parents])
                parented_activation = _dense_layer(parented_activation, layers, tm.annotation_units, activation, conv_normalize)
                output_predictions[tm.output_name()] = Dense(units=tm.shape[0], activation=tm.activation, name=tm.output_name())(parented_activation)
        elif tm.is_categorical_any():
            output_predictions[tm.output_name()] = Dense(units=tm.shape[0], activation='softmax', name=tm.output_name())(multimodal_activation)
        else:
            output_predictions[tm.output_name()] = Dense(units=tm.shape[0], activation=tm.activation, name=tm.output_name())(multimodal_activation)

    m = Model(inputs=input_tensors, outputs=list(output_predictions.values()))
    m.summary()

    model_layers = kwargs.get('model_layers', False)
    if model_layers:
        loaded = 0
        freeze =  kwargs.get('freeze_model_layers', False)
        m_other = load_model(model_layers, custom_objects=custom_dict)
        for l in m_other.layers:
            try:
                target_layer = m.get_layer(l.name)
                target_layer.set_weights(l.get_weights())
                loaded += 1
                if freeze:
                    target_layer.trainable = False
            except (ValueError, KeyError):
                continue
        logging.info(f'Loaded {"and froze " if freeze else ""}{loaded} layers from {model_layers}.')

    m.compile(optimizer=opt, loss=losses, loss_weights=loss_weights, metrics=my_metrics)
    return m


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~ Training ~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def train_model_from_generators(model: Model,
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
                                return_history: bool = False,
                                plot: bool = True) -> Union[Model, Tuple[Model, History]]:
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
    :param return_history: If true return history from training and don't plot the training history
    :return: The optimized model.
    """
    model_file = os.path.join(output_folder, run_id, run_id + TENSOR_EXT)
    if not os.path.exists(os.path.dirname(model_file)):
        os.makedirs(os.path.dirname(model_file))

    if inspect_model:
        image_p = os.path.join(output_folder, run_id, 'architecture_graph_' + run_id + IMAGE_EXT)
        _inspect_model(model, generate_train, generate_valid, batch_size, training_steps, inspect_show_labels, image_p)

    history = model.fit_generator(generate_train, steps_per_epoch=training_steps, epochs=epochs, verbose=1,
                                  validation_steps=validation_steps, validation_data=generate_valid,
                                  callbacks=_get_callbacks(patience, model_file),)

    logging.info('Model weights saved at: %s' % model_file)
    if plot:
        plot_metric_history(history, run_id, os.path.dirname(model_file))
    if return_history:
        return model, history
    return model


def _get_callbacks(patience: int, model_file: str) -> List[Callable]:
    callbacks = [
        ModelCheckpoint(filepath=model_file, verbose=1, save_best_only=True),
        EarlyStopping(monitor='val_loss', patience=patience * 3, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=patience, verbose=1)
    ]

    return callbacks


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~ Predicting ~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def embed_model_predict(model, tensor_maps_in, embed_layer, test_data, batch_size):
    embed_model = make_hidden_layer_model(model, tensor_maps_in, embed_layer)
    return embed_model.predict(test_data, batch_size=batch_size)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~ Model Builders ~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def _conv_block_new(x: K.placeholder,
                    layers: Dict[str, K.placeholder],
                    conv_layers: List[Layer],
                    pool_layers: List[Layer],
                    dimension: int,
                    activation: str,
                    normalization: str,
                    regularization: str,
                    regularization_rate: float,
                    residual_convolution_layer: Layer):
    pool_diff = len(conv_layers) - len(pool_layers)

    for i, conv_layer in enumerate(conv_layers):
        residual = x
        x = layers[f"Conv_{str(len(layers))}"] = conv_layer(x)
        x = layers[f"Activation_{str(len(layers))}"] = _activation_layer(activation)(x)
        x = layers[f"Normalization_{str(len(layers))}"] = _normalization_layer(normalization)(x)
        x = layers[f"Regularization_{str(len(layers))}"] = _regularization_layer(dimension, regularization, regularization_rate)(x)
        if i >= pool_diff:
            x = layers[f"Pooling_{str(len(layers))}"] = pool_layers[i - pool_diff](x)
            if residual_convolution_layer is not None:
                residual = layers[f"Pooling_{str(len(layers))}"] = pool_layers[i - pool_diff](residual)
        if residual_convolution_layer is not None:
            if K.int_shape(x)[CHANNEL_AXIS] == K.int_shape(residual)[CHANNEL_AXIS]:
                x = layers[f"add_{str(len(layers))}"] = add([x, residual])
            else:
                residual = layers[f"Conv_{str(len(layers))}"] = residual_convolution_layer(filters=K.int_shape(x)[CHANNEL_AXIS], kernel_size=(1, 1))(residual)
                x = layers[f"add_{str(len(layers))}"] = add([x, residual])
    return _get_last_layer(layers)


def _dense_block(x: K.placeholder,
                 layers: Dict[str, K.placeholder],
                 block_size: int,
                 conv_layers: List[Layer],
                 pool_layers: List[Layer],
                 dimension: int,
                 activation: str,
                 normalization: str,
                 regularization: str,
                 regularization_rate: float):
    for i, conv_layer in enumerate(conv_layers):
        x = layers[f"Conv_{str(len(layers))}"] = conv_layer(x)
        x = layers[f"Activation_{str(len(layers))}"] = _activation_layer(activation)(x)
        x = layers[f"Normalization_{str(len(layers))}"] = _normalization_layer(normalization)(x)
        x = layers[f"Regularization_{str(len(layers))}"] = _regularization_layer(dimension, regularization, regularization_rate)(x)
        if i % block_size == 0:  # TODO: pools should come AFTER the dense conv block not before.
            x = layers[f"Pooling{JOIN_CHAR}{str(len(layers))}"] = pool_layers[i//block_size](x)
            dense_connections = [x]
        else:
            dense_connections += [x]
            x = layers[f"concatenate{JOIN_CHAR}{str(len(layers))}"] = concatenate(dense_connections, axis=CHANNEL_AXIS)
    return _get_last_layer(layers)


def _one_by_n_kernel(dimension):
    return tuple([1] * (dimension-1))


def _conv_layer_from_kind_and_dimension(dimension, conv_layer_type, conv_width, conv_x, conv_y, conv_z):
    if dimension == 4 and conv_layer_type == 'conv':
        conv_layer = Conv3D
        kernel = (conv_x, conv_y, conv_z)
    elif dimension == 3 and conv_layer_type == 'conv':
        conv_layer = Conv2D
        kernel = (conv_x, conv_y)
    elif dimension == 2 and conv_layer_type == 'conv':
        conv_layer = Conv1D
        kernel = conv_width
    elif dimension == 3 and conv_layer_type == 'separable':
        conv_layer = SeparableConv2D
        kernel = (conv_x, conv_y)
    elif dimension == 2 and conv_layer_type == 'separable':
        conv_layer = SeparableConv1D
        kernel = conv_width
    elif dimension == 3 and conv_layer_type == 'depth':
        conv_layer = DepthwiseConv2D
        kernel = (conv_x, conv_y)
    else:
        raise ValueError(f'Unknown convolution type: {conv_layer_type} for dimension: {dimension}')
    return conv_layer, kernel


def _conv_layers_from_kind_and_dimension(dimension, conv_layer_type, conv_layers, conv_width, conv_x, conv_y, conv_z, padding, dilate, inner_loop=1):
    conv_layer, kernel = _conv_layer_from_kind_and_dimension(dimension, conv_layer_type, conv_width, conv_x, conv_y, conv_z)
    dilation_rate = 1
    conv_layer_functions = []
    for i, c in enumerate(conv_layers):
        for _ in range(inner_loop):
            if dilate:
                dilation_rate = 1 << i
            conv_layer_functions.append(conv_layer(filters=c, kernel_size=kernel, padding=padding, dilation_rate=dilation_rate))

    return conv_layer_functions


def _pool_layers_from_kind_and_dimension(dimension, pool_type, pool_number, pool_x, pool_y, pool_z):
    if dimension == 4 and pool_type == 'max':
        return [MaxPooling3D(pool_size=(pool_x, pool_y, pool_z)) for _ in range(pool_number)]
    elif dimension == 3 and pool_type == 'max':
        return [MaxPooling2D(pool_size=(pool_x, pool_y)) for _ in range(pool_number)]
    elif dimension == 2 and pool_type == 'max':
        return [MaxPooling1D(pool_size=pool_x) for _ in range(pool_number)]
    elif dimension == 4 and pool_type == 'average':
        return [AveragePooling3D(pool_size=(pool_x, pool_y, pool_z)) for _ in range(pool_number)]
    elif dimension == 3 and pool_type == 'average':
        return [AveragePooling2D(pool_size=(pool_x, pool_y)) for _ in range(pool_number)]
    elif dimension == 2 and pool_type == 'average':
        return [AveragePooling1D(pool_size=pool_x) for _ in range(pool_number)]
    else:
        raise ValueError(f'Unknown pooling type: {pool_type} for dimension: {dimension}')


def _dense_layer(x: K.placeholder, layers: Dict[str, K.placeholder], units: int, activation: str, normalization: str, name=None):
        if name is not None:
            x = layers[f"{name}_{str(len(layers))}"] = Dense(units=units, name=name)(x)
        else:
            x = layers[f"Dense_{str(len(layers))}"] = Dense(units=units)(x)
        x = layers[f"Activation_{str(len(layers))}"] = _activation_layer(activation)(x)
        x = layers[f"Normalization_{str(len(layers))}"] = _normalization_layer(normalization)(x)
        return _get_last_layer(layers)


def _upsampler(dimension, pool_x, pool_y, pool_z):
    if dimension == 4:
        return UpSampling3D(size=(pool_x, pool_y, pool_z))
    elif dimension == 3 :
        return UpSampling2D(size=(pool_x, pool_y))
    elif dimension == 2:
        return UpSampling1D(size=pool_x)


def _activation_layer(activation):
    if activation == 'leaky':
        return LeakyReLU()
    elif activation == 'prelu':
        return PReLU()
    elif activation == 'elu':
        return ELU()
    elif activation == 'thresh_relu':
        return ThresholdedReLU()
    else:
        return Activation(activation)


def _normalization_layer(norm):
    if norm == 'batch_norm':
        return BatchNormalization(axis=CHANNEL_AXIS)
    else:
        return lambda x: x


def _regularization_layer(dimension, regularization_type, rate):
    if dimension == 4 and regularization_type == 'spatial_dropout':
        return SpatialDropout3D(rate)
    elif dimension == 3 and regularization_type == 'spatial_dropout':
        return SpatialDropout2D(rate)
    elif dimension == 2 and regularization_type == 'spatial_dropout':
        return SpatialDropout1D(rate)
    elif regularization_type == 'dropout':
        return Dropout(rate)
    else:
        return lambda x: x


def _get_last_layer(named_layers):
    max_index = -1
    max_layer = ''
    for k in named_layers:
        cur_index = int(k.split('_')[-1])
        if max_index < cur_index:
            max_index = cur_index
            max_layer = k
    return named_layers[max_layer]


def _get_last_layer_by_kind(named_layers, kind, mask_after=9e9):
    max_index = -1
    for k in named_layers:
        if kind in k:
            val = int(k.split('_')[-1])
            if val < mask_after:
                max_index = max(max_index, val)
    return named_layers[kind + JOIN_CHAR + str(max_index)]


def _get_layer_kind_sorted(named_layers, kind):
    return [k for k in sorted(list(named_layers.keys()), key=lambda x: int(x.split('_')[-1])) if kind in k]


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~ Inspections ~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def _inspect_model(model: Model,
                  generate_train: Iterable,
                  generate_valid: Iterable,
                  batch_size: int,
                  training_steps: int,
                  inspect_show_labels: bool,
                  image_path: str) -> Model:
    """Collect statistics on model inference and training times.

	Arguments
	    model: the model to inspect
		generate_train: training data generator function
		generate_valid: Validation data generator function
		batch_size: size of the mini-batches
		training_steps: number of optimization steps to take
		inspect_show_labels: if True, show layer labels on the architecture diagram
		image_path: file path of the architecture diagram

	Returns
		The slightly optimized keras model
	"""
    if image_path:
        _plot_dot_model_in_color(model_to_dot(model, show_shapes=inspect_show_labels, expand_nested=True), image_path, inspect_show_labels)

    t0 = time.time()
    _ = model.fit_generator(generate_train, steps_per_epoch=training_steps, validation_steps=1, validation_data=generate_valid)
    t1 = time.time()
    train_speed = (t1 - t0) / (batch_size * training_steps)
    logging.info('Spent:{} seconds training, batch_size:{} steps:{} Per example training speed:{}'.format(
        (t1 - t0), batch_size, training_steps, train_speed))

    t0 = time.time()
    _ = model.predict_generator(generate_valid, steps=training_steps, verbose=1)
    t1 = time.time()
    inference_speed = (t1 - t0) / (batch_size * training_steps)
    logging.info('Spent:{} seconds predicting. Per tensor inference speed:{}'.format((t1 - t0), inference_speed))

    return model


def _plot_dot_model_in_color(dot, image_path, inspect_show_labels):
    import pydot
    legend = {}
    for n in dot.get_nodes():
        if n.get_label():
            if 'Conv1' in n.get_label():
                legend['Conv1'] = "cyan"
                n.set_fillcolor("cyan")
            elif 'Conv2' in n.get_label():
                legend['Conv2'] = "deepskyblue1"
                n.set_fillcolor("deepskyblue1")
            elif 'Conv3' in n.get_label():
                legend['Conv3'] = "deepskyblue3"
                n.set_fillcolor("deepskyblue3")
            elif 'UpSampling' in n.get_label():
                legend['UpSampling'] = "darkslategray2"
                n.set_fillcolor("darkslategray2")
            elif 'Transpose' in n.get_label():
                legend['Transpose'] = "deepskyblue2"
                n.set_fillcolor("deepskyblue2")
            elif 'BatchNormalization' in n.get_label():
                legend['BatchNormalization'] = "goldenrod1"
                n.set_fillcolor("goldenrod1")
            elif 'output_' in n.get_label():
                n.set_fillcolor("darkolivegreen2")
                legend['Output'] = "darkolivegreen2"
            elif 'softmax' in n.get_label():
                n.set_fillcolor("chartreuse")
                legend['softmax'] = "chartreuse"
            elif 'MaxPooling' in n.get_label():
                legend['MaxPooling'] = "aquamarine"
                n.set_fillcolor("aquamarine")
            elif 'Dense' in n.get_label():
                legend['Dense'] = "gold"
                n.set_fillcolor("gold")
            elif 'Reshape' in n.get_label():
                legend['Reshape'] = "coral"
                n.set_fillcolor("coral")
            elif 'Input' in n.get_label():
                legend['Input'] = "darkolivegreen1"
                n.set_fillcolor("darkolivegreen1")
            elif 'Activation' in n.get_label():
                legend['Activation'] = "yellow"
                n.set_fillcolor("yellow")
        n.set_style("filled")
        if not inspect_show_labels:
            n.set_label('\n')

    for l in legend:
        legend_node = pydot.Node('legend'+l, label=l, shape="box", fillcolor=legend[l])
        dot.add_node(legend_node)

    logging.info('Saving architecture diagram to:{}'.format(image_path))
    dot.write_png(image_path)


def _get_tensor_maps_for_characters(tensor_maps_in: List[TensorMap], base_model: Model):
    embed_model = make_hidden_layer_model(base_model, tensor_maps_in, 'embed')
    tm_embed = TensorMap('embed', shape=(64,), group='hidden_layer', required_inputs=tensor_maps_in.copy(), model=embed_model)
    tm_char = TensorMap('ecg_rest_next_char', shape=(len(ECG_CHAR_2_IDX),), channel_map=ECG_CHAR_2_IDX, activation='softmax', loss='categorical_crossentropy', loss_weight=1.0, cacheable=False)
    tm_burn_in = TensorMap('ecg_rest_text', shape=(100, len(ECG_CHAR_2_IDX)), group='ecg_text', channel_map={'context': 0, 'alphabet': 1}, dependent_map=tm_char, cacheable=False)
    return [tm_embed, tm_burn_in], [tm_char]


def get_model_inputs_outputs(model_files: List[str],
                             tensor_maps_in: List[TensorMap],
                             tensor_maps_out: List[TensorMap]) -> Dict[str, Dict[str, TensorMap]]:
    """Organizes given input and output tensors as nested dictionary.

    Returns:
        dict: The nested dictionary of tensors.
            The inner dictionary is keyed by tensor type ('input' or 'output').
            The outer dictionary is keyed by 'model_file'.

            {
                'model_file_1':
                    {
                        'input': [tensor1, tensor2],
                        'output': [tensor3, tensor4]
                    },
                'model_file_2':
                    {
                        'input': [tensor2, tensor5],
                        'output': [tensor4, tensor6]
                    }
            }
                        
    """

    input_prefix = "input"
    output_prefix = "output"
    got_tensor_maps_for_characters = False
    models_inputs_outputs = dict()

    for model_file in model_files:
        with h5py.File(model_file, 'r') as hd5:
            model_inputs_outputs = defaultdict(list)
            for input_tensor_map in tensor_maps_in:
                if input_tensor_map.input_name() in hd5["model_weights"]:
                    model_inputs_outputs[input_prefix].append(input_tensor_map)
            for output_tensor_map in tensor_maps_out:
                if output_tensor_map.output_name() in hd5["model_weights"]:
                    model_inputs_outputs[output_prefix].append(output_tensor_map)
            if not got_tensor_maps_for_characters and 'input_ecg_rest_text_ecg_text' in hd5["model_weights"]:
                m = load_model(model_file, custom_objects=get_metric_dict(tensor_maps_out))
                char_maps_in, char_maps_out = _get_tensor_maps_for_characters(tensor_maps_in, m)
                model_inputs_outputs[input_prefix].extend(char_maps_in)
                tensor_maps_in.extend(char_maps_in)
                model_inputs_outputs[output_prefix].extend(char_maps_out)
                tensor_maps_out.extend(char_maps_out)
                got_tensor_maps_for_characters = True
                logging.info(f"Doing char model dance:{[tm.input_name() for tm in tensor_maps_in]}")
                logging.info(f"Doing char model dance out:{[tm.output_name() for tm in tensor_maps_out]}")

        models_inputs_outputs[model_file] = model_inputs_outputs

    return models_inputs_outputs


