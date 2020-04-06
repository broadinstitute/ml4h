import os
import pytest
import numpy as np
import tensorflow as tf
from itertools import cycle
from collections import defaultdict
from typing import List, Optional, Dict, Tuple, Iterator

from ml4cvd.models import make_multimodal_multitask_model, parent_sort
from ml4cvd.TensorMap import TensorMap
from ml4cvd.test_utils import TMAPS_UP_TO_4D, MULTIMODAL_UP_TO_4D, CONTINUOUS_TMAPS, SEGMENT_IN, SEGMENT_OUT, PARENT_TMAPS, CYCLE_PARENTS


DEFAULT_PARAMS = {
    'activation': 'relu',
    'dense_layers': [4, 2],
    'dense_blocks': [5, 3],
    'block_size': 3,
    'conv_width': 3,
    'learning_rate': 1e-3,
    'optimizer': 'adam',
    'conv_type': 'conv',
    'conv_layers': [6, 5, 3],
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
    'bottleneck_type': 'flatten_restructure',
}


TrainType = Dict[str, np.ndarray]  # TODO: better name


def make_training_data(input_tmaps: List[TensorMap], output_tmaps: List[TensorMap]) -> Iterator[Tuple[TrainType, TrainType, List[None]]]:
    return cycle([
        (
            {tm.input_name(): tf.random.normal((2,) + tm.shape) for tm in input_tmaps},
            {tm.output_name(): tf.zeros((2,) + tm.shape) for tm in output_tmaps},
            [None] * len(output_tmaps),
        ), ])


def assert_model_trains(input_tmaps: List[TensorMap], output_tmaps: List[TensorMap], m: Optional[tf.keras.Model] = None):
    if m is None:
        m = make_multimodal_multitask_model(
            input_tmaps,
            output_tmaps,
            **DEFAULT_PARAMS,
        )
    for tmap, tensor in zip(input_tmaps, m.inputs):
        assert tensor.shape[1:] == tmap.shape
        assert tensor.shape[1:] == tmap.shape
    for tmap, tensor in zip(parent_sort(output_tmaps), m.outputs):
        assert tensor.shape[1:] == tmap.shape
        assert tensor.shape[1:] == tmap.shape
    data = make_training_data(input_tmaps, output_tmaps)
    history = m.fit(data, steps_per_epoch=2, epochs=2, validation_data=data, validation_steps=2)
    for tmap in output_tmaps:
        for metric in tmap.metrics:
            metric_name = metric if type(metric) == str else metric.__name__
            name = f'{tmap.output_name()}_{metric_name}' if len(output_tmaps) > 1 else metric_name
            assert name in history.history


def _rotate(a: List, n: int):
    return a[-n:] + a[:-n]


class TestMakeMultimodalMultitaskModel:
    @pytest.mark.parametrize(
        'input_output_tmaps',
        [
            (CONTINUOUS_TMAPS[:1], CONTINUOUS_TMAPS[1:2]), (CONTINUOUS_TMAPS[1:2], CONTINUOUS_TMAPS[:1]),
            (CONTINUOUS_TMAPS[:2], CONTINUOUS_TMAPS[:2]),
        ],
    )
    def test_multimodal_multitask_quickly(self, input_output_tmaps):
        """
        Tests 1d->2d, 2d->1d, (1d,2d)->(1d,2d)
        """
        assert_model_trains(input_output_tmaps[0], input_output_tmaps[1])

    @pytest.mark.slow
    @pytest.mark.parametrize(
        'input_tmaps',
        MULTIMODAL_UP_TO_4D,
    )
    @pytest.mark.parametrize(
        'output_tmaps',
        MULTIMODAL_UP_TO_4D,
    )
    def test_multimodal(self, input_tmaps: List[TensorMap], output_tmaps: List[TensorMap]):
        assert_model_trains(input_tmaps, output_tmaps)

    @pytest.mark.slow
    @pytest.mark.parametrize(
        'input_tmap',
        CONTINUOUS_TMAPS[:-1],
    )
    @pytest.mark.parametrize(
        'output_tmap',
        TMAPS_UP_TO_4D,
    )
    def test_unimodal_md_to_nd(self, input_tmap: TensorMap, output_tmap: TensorMap):
        assert_model_trains([input_tmap], [output_tmap])

    @pytest.mark.slow
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

    @pytest.mark.slow
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

    def test_u_connect_auto_encode(self):
        params = DEFAULT_PARAMS.copy()
        params['pool_x'] = params['pool_y'] = 2
        params['conv_layers'] = [8, 8]
        params['dense_blocks'] = [4, 4, 2]
        m = make_multimodal_multitask_model(
            [SEGMENT_IN],
            [SEGMENT_IN],
            u_connect=defaultdict(set, {SEGMENT_IN: {SEGMENT_IN}}),
            **params,
        )
        assert_model_trains([SEGMENT_IN], [SEGMENT_IN], m)

    def test_u_connect_segment(self):
        params = DEFAULT_PARAMS.copy()
        params['pool_x'] = params['pool_y'] = 2
        m = make_multimodal_multitask_model(
            [SEGMENT_IN],
            [SEGMENT_OUT],
            u_connect=defaultdict(set, {SEGMENT_IN: {SEGMENT_OUT}}),
            **params,
        )
        assert_model_trains([SEGMENT_IN], [SEGMENT_OUT], m)

    def test_u_connect_adaptive_normalization(self):
        params = DEFAULT_PARAMS.copy()
        params['pool_x'] = params['pool_y'] = 2
        params['bottleneck_type'] = 'squeeze_excitation'
        m = make_multimodal_multitask_model(
            [SEGMENT_IN, TMAPS_UP_TO_4D[0]],
            [SEGMENT_OUT],
            u_connect=defaultdict(set, {SEGMENT_IN: {SEGMENT_OUT}}),
            **params,
        )
        assert_model_trains([SEGMENT_IN, TMAPS_UP_TO_4D[0]], [SEGMENT_OUT], m)

    @pytest.mark.parametrize(
        'output_tmaps',
        [_rotate(PARENT_TMAPS, i) for i in range(len(PARENT_TMAPS))],
    )
    def test_parents(self, output_tmaps):
        assert_model_trains([TMAPS_UP_TO_4D[-1]], output_tmaps)


@pytest.mark.parametrize(
    'tmaps',
    [_rotate(PARENT_TMAPS, i) for i in range(len(PARENT_TMAPS))],
)
def test_parent_sort(tmaps):
    assert parent_sort(tmaps) == PARENT_TMAPS


@pytest.mark.parametrize(
    'tmaps',
    [_rotate(CYCLE_PARENTS, i) for i in range(len(CYCLE_PARENTS))],
)
def test_parent_sort_cycle(tmaps):
    with pytest.raises(ValueError):
        parent_sort(tmaps)
