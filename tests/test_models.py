import pytest

import keras.backend as K

from ml4cvd.models import make_multimodal_multitask_model
from ml4cvd.TensorMap import TensorMap, Interpretation



tmap_continuous_1d = TensorMap('1d', shape=(2,), interpretation=Interpretation.CONTINUOUS)
tmap_continuous_2d = TensorMap('2d', shape=(2, 3), interpretation=Interpretation.CONTINUOUS)
tmap_continuous_3d = TensorMap('3d', shape=(2, 3, 4), interpretation=Interpretation.CONTINUOUS)
tmap_continuous_4d = TensorMap('4d', shape=(2, 3, 4, 5), interpretation=Interpretation.CONTINUOUS)

tmap_categorical_1d = TensorMap('1d_cat', channel_map={}, shape=(2,), interpretation=Interpretation.CATEGORICAL)
tmap_categorical_2d = TensorMap('2d_cat', channel_map={}, shape=(2, 3), interpretation=Interpretation.CATEGORICAL)
tmap_categorical_3d = TensorMap('3d_cat', channel_map={}, shape=(2, 3, 4), interpretation=Interpretation.CATEGORICAL)
tmap_categorical_4d = TensorMap('4d_cat', channel_map={}, shape=(2, 3, 4, 5), interpretation=Interpretation.CATEGORICAL)
tmap_categorical_5d = TensorMap('5d_cat', channel_map={}, shape=(2, 3, 4, 5, 6), interpretation=Interpretation.CATEGORICAL)

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
}


def layer_shape(layer):
    shape = K.int_shape(layer)
    if shape[0] is None:
        shape = shape[1:]
    return shape


class TestMakeMultimodalMultitaskModel:

    @pytest.mark.parametrize(
        'input_tmap',
        CATEGORICAL_TMAPS[:-1] + CONTINUOUS_TMAPS[:-1],
    )
    @pytest.mark.parametrize(
        'output_tmap',
        CATEGORICAL_TMAPS[:1] + CONTINUOUS_TMAPS[:1],
    )
    def test_unimodal_1d_task(self, input_tmap: TensorMap, output_tmap: TensorMap):
        m = make_multimodal_multitask_model(
            [input_tmap],
            [output_tmap],
            **DEFAULT_PARAMS,
        )
        assert m.input_shape[1:] == input_tmap.shape
        assert m.output_shape[1:] == output_tmap.shape
        assert m.input_names[0] == input_tmap.input_name()
        assert m.output_names[0] == output_tmap.output_name()

    @pytest.mark.parametrize(
        'input_tmap',
        CATEGORICAL_TMAPS[-1:] + CONTINUOUS_TMAPS[-1:],
    )
    @pytest.mark.parametrize(
        'output_tmap',
        CATEGORICAL_TMAPS[:-1] + CONTINUOUS_TMAPS[:-1],
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
        CATEGORICAL_TMAPS[:-1] + CONTINUOUS_TMAPS[:-1],
    )
    @pytest.mark.parametrize(
        'output_tmap',
        CATEGORICAL_TMAPS[-1:] + CONTINUOUS_TMAPS[-1:],
    )
    def test_output_too_high_dimensional(self, input_tmap, output_tmap):
        with pytest.raises(ValueError):
            make_multimodal_multitask_model(
                [input_tmap],
                [output_tmap],
                **DEFAULT_PARAMS,
            )

    @pytest.mark.parametrize(
        'input_tmap',
        CATEGORICAL_TMAPS[:1] + CONTINUOUS_TMAPS[:1],
    )
    @pytest.mark.parametrize(
        'output_tmap',
        CATEGORICAL_TMAPS[-1:] + CONTINUOUS_TMAPS[-1:],
    )
    def test_1d_to_nd(self, input_tmap, output_tmap):
        """
        This is a test we would like to pass, but fails now
        """
        with pytest.raises(UnboundLocalError):
            make_multimodal_multitask_model(
                [input_tmap],
                [output_tmap],
                **DEFAULT_PARAMS,
            )
