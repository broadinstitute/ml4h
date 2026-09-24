"""Training-run artifacts rebuild the same heads for fine-tuning and inference."""

import json
from pathlib import Path
import sys

import numpy as np
import pytest
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model_descriptions.droid_model import build_model, head_spec, load_trained_model, read_trained_run

FRAMES = 2
SURVIVAL_TASK = dict(name='af', event_column='af_status', follow_up_days_column='af_time',
                     intervals=5, days_window=1825)


def small_encoder():
    inputs = tf.keras.Input((FRAMES, 224, 224, 3))
    features = tf.keras.layers.GlobalAveragePooling3D()(inputs)
    return tf.keras.Model(inputs, tf.keras.layers.Dense(8)(features))


def write_run(root, output_labels, output_labels_types, cls_map=None, survival=()):
    run = root / 'run'
    (run / 'model').mkdir(parents=True)
    params = {'output_labels': output_labels, 'output_labels_types': output_labels_types,
              'survival_task': list(survival), 'n_input_frames': FRAMES, 'skip_modulo': 2}
    (run / 'model_params.json').write_text(json.dumps(params))
    if cls_map:
        (run / 'classification_class_label_mapping_per_output.json').write_text(json.dumps(cls_map))
    return str(run / 'model' / 'chkp')


@pytest.mark.parametrize('labels,types,cls_map,survival', [
    (['LVEF'], 'r', None, ()),
    (['LVEF', 'sex'], 'rc', {'sex': {'0': 0, '1': 1}, 'cls_output_order': ['sex']}, ()),
    (None, 'r', None, (SURVIVAL_TASK,)),
    (['sex', 'LVEF'], 'cr', {'sex': {'0': 0, '1': 1}, 'cls_output_order': ['sex']}, (SURVIVAL_TASK,)),
])
def test_round_trip(tmp_path, labels, types, cls_map, survival):
    checkpoint = write_run(tmp_path, labels, types, cls_map, survival)
    run = read_trained_run(checkpoint)
    assert run.head_spec == head_spec(run.n_regression, cls_map, list(survival))
    trained_encoder = small_encoder()
    trained = build_model(trained_encoder, run.head_spec, FRAMES)
    trained.save_weights(checkpoint)

    encoder = small_encoder()
    loaded, _ = load_trained_model(checkpoint, encoder, FRAMES, trainable=False)
    assert loaded.output_names == trained.output_names
    if survival:
        assert loaded.output_names[-1] == 'survival_af'
    clips = np.random.default_rng(0).random((3, FRAMES, 224, 224, 3), dtype=np.float32)
    for expected, actual in zip(tf.nest.flatten(trained(clips)), tf.nest.flatten(loaded(clips))):
        np.testing.assert_allclose(expected, actual, rtol=1e-6)
    # The encoder passed in carries the trained weights, so embeddings match too.
    np.testing.assert_allclose(trained_encoder(clips), encoder(clips), rtol=1e-6)


def test_run_without_outputs_is_rejected(tmp_path):
    with pytest.raises(ValueError, match='records no'):
        read_trained_run(write_run(tmp_path, None, 'r'))
