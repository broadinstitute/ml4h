"""Lightweight tests for the DROID backbone registry and embedding contract."""

import os
import sys

import pytest

tf = pytest.importorskip('tensorflow')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model_descriptions import backbones  # noqa: E402


def _dummy_builder(*, n_input_frames, **kwargs):
    del kwargs
    inputs = tf.keras.Input(shape=(n_input_frames, 4, 4, 3), name='image')
    features = tf.keras.layers.GlobalAveragePooling3D()(inputs)
    return tf.keras.Model(inputs=inputs, outputs=features)


def test_backbone_aliases_are_canonicalized():
    assert backbones.canonical_backbone_name('movinet') == backbones.MOVINET_A2
    assert backbones.canonical_backbone_name('vjepa2_1') == backbones.VJEPA2_1_VIT_BASE_384


def test_registry_applies_requested_embedding_projection(monkeypatch):
    monkeypatch.setitem(backbones.BACKBONE_BUILDERS, 'dummy', _dummy_builder)

    encoder = backbones.create_video_encoder(
        backbone_name='dummy',
        n_input_frames=8,
        batch_size=2,
        checkpoint_path=None,
        embedding_dim=17,
    )

    assert encoder.output_shape == (None, 17)
    assert encoder.get_layer('embedding_projection').trainable


def test_embedding_dimension_must_be_positive(monkeypatch):
    monkeypatch.setitem(backbones.BACKBONE_BUILDERS, 'dummy', _dummy_builder)

    with pytest.raises(ValueError, match='embedding_dim'):
        backbones.create_video_encoder(
            backbone_name='dummy',
            n_input_frames=8,
            batch_size=2,
            checkpoint_path=None,
            embedding_dim=0,
        )


def test_vjepa_checkpoint_prefixes_are_removed():
    state_dict = backbones.FrozenVJEPA2Encoder._encoder_state_dict({
        'ema_encoder': {
            'module.backbone.blocks.0.weight': object(),
        },
    })

    assert list(state_dict) == ['blocks.0.weight']
