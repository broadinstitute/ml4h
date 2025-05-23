# The Model factory connects input TensorMaps to output TensorMaps with computational graphs.

# Imports
import os
import logging
from itertools import chain
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Set, DefaultDict, Any, Union

import tensorflow as tf

import keras

from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Layer, Lambda
from tensorflow_hub import KerasLayer

from ml4h.models.Block import Block
from ml4h.TensorMap import TensorMap
from ml4h.metrics import get_metric_dict

from ml4h.optimizers import get_optimizer

from ml4h.models.perceiver_blocks import PerceiverEncoder, PerceiverLatentLayer
from ml4h.models.layer_wrappers import ACTIVATION_FUNCTIONS, NORMALIZATION_CLASSES
from ml4h.models.pretrained_blocks import ResNetEncoder, MoviNetEncoder, BertEncoder
from ml4h.models.transformer_blocks_embedding import TransformerEncoderEmbedding,MultiHeadAttention
from ml4h.models.transformer_blocks import TransformerDecoder, TransformerEncoder, PositionalEncoding
from ml4h.models.basic_blocks import LinearDecoder, PartitionedLinearDecoder, LanguagePredictionBlock, RandomGauss, \
    IdentityEncoderBlock, IdentityDecoderBlock
from ml4h.models.basic_blocks import ModelAsBlock, LSTMEncoderBlock, LanguageDecoderBlock, DenseEncoder, DenseDecoder

from ml4h.models.merge_blocks import GlobalAveragePoolBlock, EncodeIdentityBlock, L2LossLayer, CosineLossLayer, \
    VariationalDiagNormal, KroneckerBlock, DropoutBlock

from ml4h.models.merge_blocks import FlatConcatDenseBlock, FlatConcatBlock, AverageBlock, PairLossBlock, ReduceMean, ContrastiveLossLayer
from ml4h.models.conv_blocks import ConvEncoderBlock, ConvEncoderMergeBlock, ConvDecoderBlock, ConvUnetDecoderBlock, ResidualBlock, PoolBlock, ConvUp, ConvDown


BLOCK_CLASSES = {
    'conv_encode': ConvEncoderBlock,
    'merge_conv_encode': ConvEncoderMergeBlock,
    'conv_decode': ConvDecoderBlock,
    'unet_conv_decode': ConvUnetDecoderBlock,
    'conv_up': ConvUp,
    'conv_down': ConvDown,
    'residual': ResidualBlock,
    'pool': PoolBlock,
    'concat': FlatConcatDenseBlock,
    'kronecker': KroneckerBlock,
    'dropout': DropoutBlock,
    'flat': FlatConcatBlock,
    'average': AverageBlock,
    'mean': ReduceMean,
    'pair': PairLossBlock,
    'gap': GlobalAveragePoolBlock,
    'lstm_encode': LSTMEncoderBlock,
    'language_decode': LanguageDecoderBlock,
    'language_predict': LanguagePredictionBlock,
    'dense_encode': DenseEncoder,
    'dense_decode': DenseDecoder,
    'linear_decode': LinearDecoder,
    'partitioned_linear_decode': PartitionedLinearDecoder,
    'identity': EncodeIdentityBlock,
    'transformer_encoder': TransformerEncoder,
    'perceiver_encoder': PerceiverEncoder,
    'transformer_encoder_embedding': TransformerEncoderEmbedding,
    'transformer_decoder': TransformerDecoder,
    'resnet_encoder': ResNetEncoder,
    'movinet_encoder': MoviNetEncoder,
    'bert_encoder': BertEncoder,
    'identity_decode': IdentityDecoderBlock,
    'identity_encode': IdentityEncoderBlock,
}


def compose(f, g):
    return lambda x, y: g(f(x, y), y)


def identity(x, _):
    return x


def make_multimodal_multitask_model(
        tensor_maps_in: List[TensorMap],
        tensor_maps_out: List[TensorMap],
        encoder_blocks: List[str],
        decoder_blocks: List[str],
        merge_blocks: List[str],
        learning_rate: float,
        optimizer: str,
        u_connect: DefaultDict[TensorMap, Set[TensorMap]] = None,
        training_steps: int = None,
        learning_rate_schedule: str = None,
        **kwargs,
) -> Tuple[Model, Dict[TensorMap, Model], Dict[TensorMap, Model]]:
    """Make multi-task, multi-modal feed forward neural network for all kinds of prediction

    This model factory can be used to make networks for classification, regression, and segmentation
    The tasks attempted are given by the output TensorMaps.
    The modalities and the first layers in the architecture are determined by the input TensorMaps.
    Model architectures are specified by Blocks which can encode, merge or decode TensorMaps.

    Hyperparameters are exposed to the command line and passed through to Block constructors via **kwargs.
    Model summary printed to output

    :param tensor_maps_in: List of input TensorMaps
    :param tensor_maps_out: List of output TensorMaps
    :param encoder_blocks: One or more BLOCK_CLASS keys or hd5 model files to encode applicable input TensorMaps
    :param decoder_blocks: One or more BLOCK_CLASS keys or hd5 model files to decode applicable output TensorMaps
    :param merge_blocks: Zero or more BLOCK_CLASS keys to construct multimodal latent space or apply loss functions based on internal activations, etc..
    :param learning_rate: learning rate for optimizer
    :param u_connect: dictionary of input TensorMap -> output TensorMaps to u connect to
    :param optimizer: which optimizer to use. See optimizers.py.
    :return: a compiled keras model
    :param learning_rate_schedule: learning rate schedule to train with, e.g. triangular
    :param training_steps: How many training steps to train the model. Only needed if learning_rate_schedule given
    """
    tensor_maps_out = parent_sort(tensor_maps_out)
    u_connect: DefaultDict[TensorMap, Set[TensorMap]] = u_connect or defaultdict(set)
    custom_dict = get_custom_objects(tensor_maps_out)
    opt = get_optimizer(
        optimizer, learning_rate, steps_per_epoch=training_steps, learning_rate_schedule=learning_rate_schedule,
        optimizer_kwargs=kwargs.get('optimizer_kwargs'),
    )


    if kwargs.get('model_file', False):
        return _load_model_encoders_and_decoders(tensor_maps_in, tensor_maps_out, custom_dict, opt, kwargs['model_file'])

    full_model, encoders, decoders, merger = multimodal_multitask_model(
        tensor_maps_in, tensor_maps_out,
        encoder_blocks, decoder_blocks, merge_blocks,
        custom_dict, u_connect, **kwargs
    )
    losses = [tm.loss for tm in tensor_maps_out]
    if len(losses) == 0:
        logging.warning(f"No losses found, hoping there is contrastive loss merge block.")
        losses = None
    full_model.compile(
        optimizer=opt, loss=losses, metrics={tm.output_name(): tm.metrics for tm in tensor_maps_out},
    )
    full_model.summary()
    #full_model.summary(print_fn=logging.info, expand_nested=True)
    if kwargs.get('model_layers', False):
        full_model.load_weights(kwargs['model_layers'], by_name=True)
        logging.info(f"Loaded model weights from:{kwargs['model_layers']}")

    return full_model, encoders, decoders, merger


def multimodal_multitask_model(
        tensor_maps_in: List[TensorMap],
        tensor_maps_out: List[TensorMap],
        encoder_blocks: List[Union[Block, str]],
        decoder_blocks: List[Union[Block, str]],
        merge_blocks: List[Union[Block, str]],
        custom_dict: Dict[str, Any],
        u_connect: DefaultDict[TensorMap, Set[TensorMap]] = None,
        **kwargs
) -> Tuple[Model, Dict[TensorMap, Model], Dict[TensorMap, Model]]:
    """Make multi-task, multi-modal feed forward neural network for all kinds of prediction

    This model factory can be used to make networks for classification, regression, and segmentation
    The tasks attempted are given by the output TensorMaps.
    The modalities and the first layers in the architecture are determined by the input TensorMaps.
    Model architectures are specified by Blocks which can encode, merge or decode TensorMaps.

    Hyperparameters are exposed to the command line and passed through to Block constructors via **kwargs.
    Model summary printed to output

    :param tensor_maps_in: List of input TensorMaps
    :param tensor_maps_out: List of output TensorMaps
    :param encoder_blocks: One or more Blocks, BLOCK_CLASS keys or hd5 model files to encode applicable input TensorMaps
    :param decoder_blocks: One or more Blocks, BLOCK_CLASS keys or hd5 model files to decode applicable output TensorMaps
    :param merge_blocks: Zero or more Blocks, BLOCK_CLASS keys to construct multimodal latent space or apply loss functions based on internal activations, etc..
    :param custom_dict: Dictionary of custom objects needed to load a serialized model
    :param u_connect: dictionary of input TensorMap -> output TensorMaps to u connect to
    """
    encoder_block_functions = {tm: identity for tm in tensor_maps_in}  # Dict[TensorMap, Block]
    for tm in tensor_maps_in:
        logging.info(f"*** Building Encoder of Tensor Map {tm} of {len(tensor_maps_in)} input TensorMaps.")

        for encode_block in encoder_blocks:  # TODO: just check that it is a block?,
            if isinstance(encode_block, Block):
                encoder_block_functions[tm] = compose(encoder_block_functions[tm], encode_block(tensor_map=tm, **kwargs))
            elif encode_block in BLOCK_CLASSES:
                encoder_block_functions[tm] = compose(encoder_block_functions[tm], BLOCK_CLASSES[encode_block](tensor_map=tm, **kwargs))

            elif encode_block.endswith(f'encoder_{tm.name}.keras'):  # TODO: load protobuf models too
                serialized_encoder = load_model(encode_block, custom_objects=custom_dict, compile=False)
                serialized_encoder = add_prefix(serialized_encoder, f'encode_block_{tm.name}', custom_dict)
                encoder_block_functions[tm] = compose(encoder_block_functions[tm], ModelAsBlock(tensor_map=tm, model=serialized_encoder))
                break  # Don't also reconstruct from scratch if model is serialized, hd5 models must precede BLOCK_CLASS keys
            else:
                logging.warning(f'{tm.name} is ignoring encoding block {encode_block}.')
    merge = identity
    for merge_block in merge_blocks:
        if isinstance(merge_block, Block):
            merge = compose(merge, merge_block(**kwargs))
        else:
            merge = compose(merge, BLOCK_CLASSES[merge_block](**kwargs))

    decoder_block_functions = {tm: identity for tm in tensor_maps_out}
    for tm in parent_sort(tensor_maps_out):
        for decode_block in decoder_blocks:
            if isinstance(decode_block, Block):
                decoder_block_functions[tm] = compose(
                    decoder_block_functions[tm], decode_block(
                        tensor_map=tm,
                        u_connect_parents=[tm_in for tm_in in tensor_maps_in if tm in u_connect[tm_in]],
                        parents=tm.parents,
                        **kwargs,
                    ),
                )
            elif decode_block in BLOCK_CLASSES:
                decoder_block_functions[tm] = compose(
                    decoder_block_functions[tm], BLOCK_CLASSES[decode_block](
                        tensor_map=tm,
                        u_connect_parents=[tm_in for tm_in in tensor_maps_in if tm in u_connect[tm_in]],
                        parents=tm.parents,
                        **kwargs,
                    ),
                )
            elif decode_block.endswith(f'decoder_{tm.name}.keras'):
                serialized_decoder = load_model(decode_block, custom_objects=custom_dict, compile=False)
                serialized_decoder = add_prefix(serialized_decoder, f'decode_block_{tm.name}', custom_dict)
                decoder_block_functions[tm] = compose(decoder_block_functions[tm], ModelAsBlock(tensor_map=tm, model=serialized_decoder))
                break
            else:
                logging.warning(f'{tm.name} is ignoring decoding block {decode_block}.')

    return make_multimodal_multitask_model_block(encoder_block_functions, merge, decoder_block_functions, u_connect, **kwargs)


def make_multimodal_multitask_model_block(
        encoder_block_functions: Dict[TensorMap, Block],
        merge: Block,
        decoder_block_functions: Dict[TensorMap, Block],  # Assumed to be topologically sorted according to parents hierarchy
        u_connect: DefaultDict[TensorMap, Set[TensorMap]],
        **kwargs
) -> Tuple[Model, Dict[TensorMap, Model], Dict[TensorMap, Model], Model]:
    """
    Turn Blocks into Models
    Returns full model, encoders, decoders and a merge model.
    Blocks are specified by dictionaries for encodings, decodings and merging.
    This function returns Models, the full model the merge model and dictionaries mapping TensorMaps to models for each encoder and decoder.

    :param encoder_block_functions: Dictionary mapping input TensorMaps to Blocks, populate intermediate Dictionary
    :param merge: A block which may use the Dictionary of intermediates to combine or add loss to a model.
    :param decoder_block_functions: Dictionary mapping output TensorMaps to Blocks, may link with intermediates for parenting or skip connections
    :param u_connect: Dictionary with TensorMap keys and sets of all their u_connected parents as values

    :return: Full model, encoder models, decoder models and merge model
    """
    inputs: Dict[TensorMap, Input] = {}
    encoders: Dict[TensorMap, Model] = {}
    encodings: List[Layer] = []
    encodings_as_inputs: List[Input] = []
    intermediates: Dict[TensorMap, List[Layer]] = defaultdict(list)

    tensor_maps_in = list(encoder_block_functions.keys())
    for tm in tensor_maps_in:
        if tm.is_text():
            inputs[tm] = Input(shape=(), dtype=tf.string, name=tm.name)
        else:
            inputs[tm] = Input(shape=tm.shape, name=tm.input_name())
        encoding = encoder_block_functions[tm](inputs[tm], intermediates)
        encoders[tm] = Model(inputs[tm], encoding, name=f'encode_{tm.name}')
        encodings.append(encoders[tm](inputs[tm]))
    

    multimodal_activation = merge(encodings, intermediates)
    merge_model = Model(list(inputs.values()), multimodal_activation)

    if isinstance(multimodal_activation, list):
        latent_inputs = Input(shape=(multimodal_activation[0].shape[-1],), name='input_multimodal_space')
    else:
        latent_inputs = Input(shape=(multimodal_activation.shape[-1],), name='input_multimodal_space')
        logging.info(f'multimodal_activation.shapes: {multimodal_activation[0].shape}')
    logging.info(f'Graph from input TensorMaps has intermediates: {[(tm, [ti.shape for ti in t]) for tm, t in intermediates.items()]}')

    tensor_maps_out = list(decoder_block_functions.keys())
    decoders: Dict[TensorMap, Model] = {}
    decoder_outputs_dict: Dict[TensorMap, Layer] = {}

    for tm in tensor_maps_out:
        if any((tm in u_connect[tm_in]) or tm.is_language() for tm_in in tensor_maps_in):
            output = decoder_block_functions[tm](multimodal_activation, intermediates)
        else:
            output = decoder_block_functions[tm](latent_inputs, intermediates)
            decoders[tm] = Model(latent_inputs, output, name=tm.output_name())
            output = decoders[tm](multimodal_activation)
        decoder_outputs_dict[tm] = output
        
    named_outputs = kwargs.get("named_outputs", False)
    if not decoder_outputs_dict:
        final_outputs = [multimodal_activation]
    else:
        if named_outputs:
            final_outputs = {
                tm.output_name(): decoder_outputs_dict[tm]
                for tm in parent_sort(tensor_maps_out, **kwargs)
            }
        else:
            final_outputs = [decoder_outputs_dict[tm] for tm in parent_sort(tensor_maps_out, **kwargs)]
    
    model_inputs = [inputs[tm] for tm in tensor_maps_in]
    full_model = Model(inputs=model_inputs, outputs=final_outputs, name='block_model')

    model_file = kwargs.get('model_file')
    if model_file:
        for tm, encoder_model in encoders.items():
            logging.info(f"saving encoder model encoder_{tm.name}.keras")
            encoder_model.save(f"{os.path.dirname(model_file)}/encoder_{tm.name}.keras")

        for tm, decoder_model in decoders.items():
            logging.info(f"saving decoder model decoder_{tm.name}.keras")
            decoder_model.save(f"{os.path.dirname(model_file)}/decoder_{tm.name}.keras")

        merge_model.save(f"{os.path.dirname(model_file)}/merger.keras")

    return full_model, encoders, decoders, merge_model


def _load_model_encoders_and_decoders(
    tensor_maps_in: List[TensorMap], tensor_maps_out: List[TensorMap], custom_dict: Dict[str, Any],
    optimizer, model_file: str,
):
    encoders = {}
    decoders = {}
    merger = None
    try:
        for tm in tensor_maps_in:
            encoders[tm] = load_model(f"{os.path.dirname(model_file)}/encoder_{tm.name}.keras", custom_objects=custom_dict, compile=False)
        for tm in tensor_maps_out:
            decoders[tm] = load_model(f"{os.path.dirname(model_file)}/decoder_{tm.name}.keras", custom_objects=custom_dict, compile=False)
        merger = load_model(f"{os.path.dirname(model_file)}/merger.keras", custom_objects=custom_dict, compile=False)
    except OSError as e:
        logging.warning(f'Could not load some model modules, error: {e}')
    logging.info(f"Attempting to load model file from: {model_file}")
    m = load_model(model_file, custom_objects=custom_dict, compile=False)
    m.compile(
        optimizer=optimizer, loss=[tm.loss for tm in tensor_maps_out],
        metrics={tm.output_name(): tm.metrics for tm in tensor_maps_out},
    )
    m.summary()
    #m.summary(print_fn=logging.info, expand_nested=True)
    logging.info(f"Loaded {len(encoders)} encoders, {len(decoders)} decoders and model file from: {model_file}")
    return m, encoders, decoders, merger


def get_custom_objects(tensor_maps_out: List[TensorMap]) -> Dict[str, Any]:
    custom_objects = {
        obj.__name__: obj
        for obj in chain(
            ACTIVATION_FUNCTIONS.values(), NORMALIZATION_CLASSES.values(),
            [
                VariationalDiagNormal, CosineLossLayer, ContrastiveLossLayer, PositionalEncoding,
                MultiHeadAttention, RandomGauss, KerasLayer, PerceiverLatentLayer, L2LossLayer,
            ],
        )
    }
    return {**custom_objects, **get_metric_dict(tensor_maps_out)}


def parent_sort(tms: List[TensorMap], **kwargs) -> List[TensorMap]:
    """
    Parents will always appear before their children after sorting. Idempotent and slow.
    """
    to_process = sorted(tms, key=lambda x: str(x))

    if kwargs.get('parent_sort', True) == False:
        return to_process
    final: List[TensorMap] = []
    visited = Counter()
    while to_process:
        tm = to_process.pop()
        visited[tm] += 1
        if visited[tm] > len(tms):
            raise ValueError('Problem detected in parent structure. Could be cycle or missing parent.')
        if not tm.parents or set(tm.parents) <= set(final):
            final.append(tm)
        else:
            to_process.insert(0, tm)
    return final


def add_prefix(model, prefix: str, custom_objects=None):
    '''Adds a prefix to layers and model name while keeping the pre-trained weights
    Arguments:
        model: a tf.keras model
        prefix: a string that would be added to before each layer name
        custom_objects: if your model consists of custom layers you shoud add them pass them as a dictionary.
            For more information read the following:
            https://keras.io/guides/serialization_and_saving/#custom-objects
    Returns:
        new_model: a tf.keras model having same weights as the input model.
    '''

    config = model.get_config()
    old_to_new = {}
    new_to_old = {}

    for layer in config['layers']:
        new_name = prefix + layer['name']
        old_to_new[layer['name']], new_to_old[new_name] = new_name, layer['name']
        layer['name'] = new_name
        layer['config']['name'] = new_name

        if len(layer['inbound_nodes']) > 0:
            for in_node in layer['inbound_nodes'][0]:
                in_node[0] = old_to_new[in_node[0]]

    for input_layer in config['input_layers']:
        input_layer[0] = old_to_new[input_layer[0]]

    for output_layer in config['output_layers']:
        output_layer[0] = old_to_new[output_layer[0]]

    config['name'] = prefix + config['name']
    new_model = tf.keras.Model().from_config(config, custom_objects)

    for layer in new_model.layers:
        layer.set_weights(model.get_layer(new_to_old[layer.name]).get_weights())

    return new_model
