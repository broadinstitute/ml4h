import os
import pytest
import tempfile
import tensorflow as tf


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


def assert_shapes_correct(input_tmap, output_tmap):
    m = make_multimodal_multitask_model(
        [input_tmap],
        [output_tmap],
        **DEFAULT_PARAMS,
    )
    assert m.input_shape[input_tmap][1:] == input_tmap.shape
    assert m.output_shape[0][1:] == output_tmap.shape
    m({input_tmap: tf.zeros((1,) + input_tmap.shape)})  # Does calling work?


class TestMakeMultimodalMultitaskModel:

    @pytest.mark.parametrize(
        'input_tmap',
        TMAPS_UP_TO_4D,
        )
    @pytest.mark.parametrize(
        'output_tmap',
        TMAPS_UP_TO_4D,
        )
    def test_unimodal_md_to_nd(self, input_tmap: TensorMap, output_tmap: TensorMap):
        assert_shapes_correct(input_tmap, output_tmap)

    @pytest.mark.parametrize(
        'input_tmap',
        TMAPS_5D,
    )
    @pytest.mark.parametrize(
        'output_tmap',
        TMAPS_UP_TO_4D,
    )
    def test_input_too_high_dimensional(self, input_tmap, output_tmap):
        with pytest.raises(ValueError):
            make_multimodal_multitask_model(
                [input_tmap],
                [output_tmap],
                **DEFAULT_PARAMS,
            )

    @pytest.mark.parametrize(
        'input_tmap',
        TMAPS_UP_TO_4D,
    )
    @pytest.mark.parametrize(
        'output_tmap',
        TMAPS_5D,
    )
    def test_output_too_high_dimensional(self, input_tmap, output_tmap):
        """
        Shows we can't handle >4d tensors.
        """
        with pytest.raises(ValueError):
            make_multimodal_multitask_model(
                [input_tmap],
                [output_tmap],
                **DEFAULT_PARAMS,
            )

    @pytest.mark.parametrize(
        'input_tmap',
        TMAPS_UP_TO_4D,
    )
    @pytest.mark.parametrize(
        'output_tmap',
        TMAPS_UP_TO_4D,
    )
    def test_load_model(self, input_tmap, output_tmap):
        m = make_multimodal_multitask_model(
            [input_tmap],
            [output_tmap],
            **DEFAULT_PARAMS,
        )
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, 'm')
            m.save(path)
            m2 = make_multimodal_multitask_model(
                [input_tmap],
                [output_tmap],
                model_file=path,
                **DEFAULT_PARAMS,
            )
