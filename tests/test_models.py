import os
import pytest
import tensorflow as tf
from typing import List
from itertools import product


from ml4cvd.models import make_multimodal_multitask_model
from ml4cvd.TensorMap import TensorMap, Interpretation


CONTINUOUS_TMAPS = [
    TensorMap(f'{n}d_cont', shape=tuple(range(1, n + 1)), interpretation=Interpretation.CONTINUOUS)
    for n in range(1, 6)
]
CATEGORICAL_TMAPS = [
    TensorMap(
        f'{n}d_cat', shape=tuple(range(1, n + 1)),
        interpretation=Interpretation.CATEGORICAL,
        channel_map={f'c_{i}': i for i in range(n)},
    )
    for n in range(1, 6)
]
TMAPS_UP_TO_4D = CONTINUOUS_TMAPS[:-1] + CATEGORICAL_TMAPS[:-1]
TMAPS_5D = CONTINUOUS_TMAPS[-1:] + CATEGORICAL_TMAPS[-1:]
MULTIMODAL_UP_TO_4D = [list(x) for x in product(CONTINUOUS_TMAPS[:-1], CATEGORICAL_TMAPS[:-1])]


DEFAULT_PARAMS = {  # TODO: should this come from the default arg parse?
    'activation': 'relu',
    'dense_layers': [4, 2],
    'dense_blocks': [5, 3],
    'block_size': 3,
    'conv_width': 3,
    'learning_rate': 1e-3,
    'optimizer': 'adam',
    'conv_type': 'conv',
    'conv_layers': [4],
    'conv_x': 3,
    'conv_y': 3,
    'conv_z': 2,
    'padding': 'same',
    'max_pools': [],
    'pool_type': 'max',
    'pool_x': 1,
    'pool_y': 1,
    'pool_z': 1,
    'dropout': 0,
    'conv_normalize': 'batch_norm',
}


def assert_shapes_correct(input_tmaps: List[TensorMap], output_tmaps: List[TensorMap]):
    m = make_multimodal_multitask_model(
        input_tmaps,
        output_tmaps,
        **DEFAULT_PARAMS,
    )
    for tmap, tensor in zip(input_tmaps, m.inputs):
        assert tensor.shape[1:] == tmap.shape
        assert tensor.shape[1:] == tmap.shape
    for tmap, tensor in zip(output_tmaps, m.outputs):
        assert tensor.shape[1:] == tmap.shape
        assert tensor.shape[1:] == tmap.shape
    m({tm.input_name(): tf.zeros((1,) + tm.shape) for tm in input_tmaps})  # Does calling work?


class TestMakeMultimodalMultitaskModel:

    @pytest.mark.parametrize(
        'input_tmaps',
        MULTIMODAL_UP_TO_4D,
    )
    @pytest.mark.parametrize(
        'output_tmaps',
        MULTIMODAL_UP_TO_4D,
    )
    def test_multimodal(self, input_tmaps: List[TensorMap], output_tmaps: List[TensorMap]):
        assert_shapes_correct(input_tmaps, output_tmaps)

    @pytest.mark.parametrize(
        'input_tmap',
        CONTINUOUS_TMAPS[:-1],
        )
    @pytest.mark.parametrize(
        'output_tmap',
        TMAPS_UP_TO_4D,
        )
    def test_unimodal_md_to_nd(self, input_tmap: TensorMap, output_tmap: TensorMap):
        assert_shapes_correct([input_tmap], [output_tmap])

    @pytest.mark.parametrize(
        'input_tmap',
        TMAPS_UP_TO_4D,
    )
    @pytest.mark.parametrize(
        'output_tmap',
        TMAPS_UP_TO_4D,
    )
    def test_load_unimodal(self, tmpdir, input_tmap, output_tmap):
        m = make_multimodal_multitask_model(
            [input_tmap],
            [output_tmap],
            **DEFAULT_PARAMS,
        )
        path = os.path.join(tmpdir, 'm')
        m.save(path)
        make_multimodal_multitask_model(
            [input_tmap],
            [output_tmap],
            model_file=path,
            **DEFAULT_PARAMS,
        )

    @pytest.mark.parametrize(
        'input_tmaps',
        MULTIMODAL_UP_TO_4D,
    )
    @pytest.mark.parametrize(
        'output_tmaps',
        MULTIMODAL_UP_TO_4D,
    )
    def test_load_multimodal(self, tmpdir, input_tmaps: List[TensorMap], output_tmaps: List[TensorMap]):
        m = make_multimodal_multitask_model(
            input_tmaps,
            output_tmaps,
            **DEFAULT_PARAMS,
        )
        path = os.path.join(tmpdir, 'm')
        m.save(path)
        make_multimodal_multitask_model(
            input_tmaps,
            output_tmaps,
            model_file=path,
            **DEFAULT_PARAMS,
        )
