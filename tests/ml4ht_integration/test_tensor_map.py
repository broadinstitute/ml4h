from itertools import cycle
from datetime import datetime, timedelta

import pytest
import numpy as np
from torch.utils.data import DataLoader
import tensorflow as tf

from ml4ht.data.data_description import DataDescription
from ml4ht.data.sample_getter import DataDescriptionSampleGetter
from ml4ht.data.data_loader import numpy_collate_fn, SampleGetterIterableDataset
from ml4ht.data.util.date_selector import DateRangeOptionPicker, first_dt, DATE_OPTION_KEY
from ml4h.ml4ht_integration.tensor_map import TensorMapSampleGetter, tensor_map_from_data_description

from ml4h.TensorMap import Interpretation
from ml4h.test_utils import TMAPS_UP_TO_4D, build_hdf5s
from ml4h.models.model_factory import make_multimodal_multitask_model, parent_sort


# Tests
def test_tensor_map_from_data_description():
    tmap_in = tensor_map_from_data_description(
        DD1_INPUT, Interpretation.CONTINUOUS, (1,), name='dd1',
    )
    tmap_out = tensor_map_from_data_description(
        DD2_OUTPUT, Interpretation.CONTINUOUS, (1,), name='dd2',
        loss='log_cosh',
        metrics=['mae', 'mse'],
    )
    model, _, _, _ = make_multimodal_multitask_model(
        [tmap_in], [tmap_out],
        encoder_blocks=['dense_encode'],
        decoder_blocks=['dense_decode'],
        merge_blocks=[],
        learning_rate=1e-4,
        optimizer='sgd',
        named_outputs = True
    )
    data_set = SampleGetterIterableDataset(
        sample_ids=list(RAW_DATA),
        sample_getter=SAMPLE_GETTER,
    )
    data_loader = DataLoader(
        data_set, batch_size=4, num_workers=2,
        collate_fn=numpy_collate_fn,
    )
    tf_dataset = tf.data.Dataset.from_generator(
        lambda : iter(data_loader),
        output_signature=(
        {'input_dd1_continuous': tf.TensorSpec(shape=(1, 1), dtype=tf.float32)},
        {'output_dd2_continuous': tf.TensorSpec(shape=(1, 1), dtype=tf.float32)}
        )
    )

    # Squeeze trailing dim so inputs/targets are (None, 1)
    def _squeeze(x, y):
        x = {'input_dd1_continuous': tf.squeeze(x['input_dd1_continuous'], axis=-1)}
        y = {'output_dd2_continuous': tf.squeeze(y['output_dd2_continuous'], axis=-1)}
        return x, y

    tf_dataset = tf_dataset.map(_squeeze)

    # model can train?
    history = model.fit(tf_dataset, epochs=1, steps_per_epoch=1).history

    # metrics recorded? (Keras may prefix with output name when named_outputs=True)
    keys = set(history.keys())
    assert ('mae' in keys) or ('output_dd2_continuous_mae' in keys)
    assert ('mse' in keys) or ('output_dd2_continuous_mse' in keys)


class TestTensorMapSampleGetter:
    def test_expected_batches(self, expected_data):
        paths = [k[0] for k in expected_data]
        sample_getter = TensorMapSampleGetter(
            TMAPS_UP_TO_4D,
            TMAPS_UP_TO_4D,
        )
        for path in paths:
            in_batch, out_batch = sample_getter(path)
            for tmap in TMAPS_UP_TO_4D:
                np.testing.assert_allclose(
                    in_batch[tmap.input_name()],
                    expected_data[path, tmap.name],
                )
                np.testing.assert_allclose(
                    out_batch[tmap.output_name()],
                    expected_data[path, tmap.name],
                )

    def test_train_model(self, expected_data, model):
        paths = [k[0] for k in expected_data]
        sample_getter = TensorMapSampleGetter(
            TMAPS_UP_TO_4D,
            TMAPS_UP_TO_4D,
        )
        dataset = SampleGetterIterableDataset(
            sample_getter=sample_getter,
            sample_ids=paths,
        )
        data_loader = DataLoader(
            dataset, batch_size=2, num_workers=3,
            collate_fn=numpy_collate_fn,
        )
        val_loader = DataLoader(
            dataset, batch_size=2, num_workers=3,
            collate_fn=numpy_collate_fn,
        )
        model_input_names = [t.name.split(":")[0] for t in model.inputs]
        tmap_output_names = [tmap.output_name() for tmap in parent_sort(TMAPS_UP_TO_4D)]
        
        def generator():
            for batch_input, batch_output in data_loader:
                inputs = [tf.convert_to_tensor(batch_input[name]) for name in model_input_names]
                outputs = [tf.convert_to_tensor(batch_output[name]) for name in tmap_output_names]
                
                yield tuple(inputs), tuple(outputs)

        example_inputs, example_outputs = next(generator())
        tf_dataset = tf.data.Dataset.from_generator(
            generator,
            output_signature=(
                tuple(tf.TensorSpec(shape=(None,) + inp.shape[1:], dtype=inp.dtype) for inp in example_inputs),
                tuple(tf.TensorSpec(shape=(None,) + out.shape[1:], dtype=out.dtype) for out in example_outputs),
            ),
        ).repeat()

        history = model.fit(
            tf_dataset, validation_data=tf_dataset,
            steps_per_epoch=10, validation_steps=2, epochs=3,
        ).history
        for tmap in TMAPS_UP_TO_4D:
            for metric in tmap.metrics:
                metric_name = metric if type(metric) == str else metric.__name__
                name = f'{tmap.output_name()}_{metric_name}'
                assert name in history
                assert f'val_{name}' in history


# Setup
@pytest.fixture(scope="module")
def expected_data(tmpdir_factory):
    temp_dir = tmpdir_factory.mktemp('data')
    return build_hdf5s(temp_dir, TMAPS_UP_TO_4D, n=pytest.N_TENSORS)


@pytest.fixture(scope="function")
def model():
    return make_multimodal_multitask_model(
        tensor_maps_in=TMAPS_UP_TO_4D,
        tensor_maps_out=TMAPS_UP_TO_4D,
        encoder_blocks=['conv_encode', 'dense_encode'],
        decoder_blocks=['conv_decode', 'dense_decode'],
        merge_blocks=['concat'],
        learning_rate=1e-4,
        optimizer='sgd',
        conv_x=[3], conv_y=[3], conv_z=[1],
        pool_x=1, pool_y=1, pool_z=1,
        dense_blocks=[4], dense_layers=[4],
        block_size=3,
        activation='relu',  conv_layers=[8], conv_type='conv',
    )[0] # Only return the full model, not encoders decoders and merger


RAW_DATA = {
    0: {
        datetime(year=2000, month=3, day=1): np.array([10]),
    },
    1: {
        datetime(year=2000, month=3, day=1): np.array([-1]),
    },
    2: {
        datetime(year=2000, month=3, day=1): np.array([10]),
    },
    3: {
        datetime(year=2000, month=3, day=1): np.array([10]),
    },
}


class DictionaryDataDescription(DataDescription):
    def __init__(self, data, fail_idx, name):
        self.data = data
        self.fail_idx = fail_idx
        self._name = name

    def get_loading_options(self, sample_id):
        return [{DATE_OPTION_KEY: dt} for dt in self.data[sample_id]]

    @property
    def name(self) -> str:
        return self._name

    def get_raw_data(self, sample_id, loading_option):
        if sample_id == self.fail_idx:
            raise ValueError("Bad idx")
        dt = loading_option[DATE_OPTION_KEY]
        return self.data[sample_id][dt]


DD1_INPUT = DictionaryDataDescription(RAW_DATA, 1, 'input_dd1_continuous')
DD2_OUTPUT = DictionaryDataDescription(RAW_DATA, 2, 'output_dd2_continuous')
DATE_OPTION_PICKER = DateRangeOptionPicker(
    reference_data_description=DD1_INPUT,
    reference_date_chooser=first_dt,
    time_before=timedelta(days=0),
    time_after=timedelta(days=5),
)
SAMPLE_GETTER = DataDescriptionSampleGetter(
    input_data_descriptions=[DD1_INPUT],
    output_data_descriptions=[DD2_OUTPUT],
    option_picker=DATE_OPTION_PICKER,
)
