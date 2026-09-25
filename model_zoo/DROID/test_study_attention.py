"""Study grouping, split isolation, and masked self-attention behavior."""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).resolve().parent))

from data_descriptions.study_embeddings import (
    assign_splits, join_embeddings_wide, make_study_dataset, make_study_records,
    smoke_test_splits,
)
from model_descriptions.droid_model import TrainedRun, head_spec, validate_survival_tasks
from model_descriptions.study_attention import build_study_attention_model


TASK = validate_survival_tasks([{
    'name': 'af', 'event_column': 'af_status', 'follow_up_days_column': 'af_time',
    'intervals': 5, 'days_window': 1825,
}])[0]


def _run():
    return TrainedRun({}, [], 0, {}, [TASK], head_spec(survival_tasks=[TASK]))


def test_study_grouping_and_patient_splits(tmp_path):
    ids = ['1_10_a', '1_10_b', '2_20_a', '3_30_a']
    wide = pd.DataFrame({
        'sample_id': ids, 'view_prediction': [0, 0, 0, 0],
        'doppler_prediction': [0] * 4, 'quality_prediction': [0] * 4,
        'canonical_prediction': [0] * 4, 'af_status': [1, 1, 0, 1],
        'af_time': [100, 100, 500, 900],
    })
    wide_path = tmp_path / 'wide.pq'
    wide.to_parquet(wide_path)
    embeddings = pd.DataFrame({
        'sample_id': ids, 'embedding_0': [1., 2., 3., 4.],
        'embedding_1': [5., 6., 7., 8.],
    })
    merged = join_embeddings_wide(embeddings, [str(wide_path)], {}, [], [TASK])
    records = make_study_records(merged, ['embedding_0', 'embedding_1'], _run())
    assert [len(record['sample_ids']) for record in records] == [2, 1, 1]
    assigned = assign_splits(records, {
        'patient_train': [1], 'patient_valid': [2], 'patient_test': [3],
    })
    assert {name: [r['study_id'] for r in group] for name, group in assigned.items()} == {
        'train': ['1_10'], 'valid': ['2_20'], 'internal_test': [], 'test': ['3_30'],
    }
    dataset = make_study_dataset(records, 2, [10], 2)
    inputs, labels = next(iter(dataset))
    assert inputs['embeddings'].shape == (2, 2, 2)
    assert inputs['mask'].numpy().tolist() == [[True, True], [True, False]]
    assert labels[0].shape == (2, 10)

    wide.loc[1, 'af_time'] = 200
    wide.to_parquet(wide_path)
    inconsistent = join_embeddings_wide(embeddings, [str(wide_path)], {}, [], [TASK])
    with pytest.raises(ValueError, match='Inconsistent labels'):
        make_study_records(inconsistent, ['embedding_0', 'embedding_1'], _run())


def test_mask_and_permutation_do_not_change_study_prediction():
    tf.keras.utils.set_random_seed(7)
    model = build_study_attention_model(4, _run().head_spec, num_heads=2, key_dim=2,
                                        dropout=0, pooling='attention')
    clips = np.array([[1., 2., 3., 4.], [4., 3., 2., 1.]], np.float32)
    plain = model({'embeddings': clips[None],
                   'mask': np.array([[True, True]])}, training=False)
    padded = model({
        'embeddings': np.concatenate([clips, [[100., 100., 100., 100.]]])[None],
        'mask': np.array([[True, True, False]]),
    }, training=False)
    permuted = model({'embeddings': clips[::-1][None],
                      'mask': np.array([[True, True]])},
                     training=False)
    np.testing.assert_allclose(plain, padded, atol=1e-5)
    np.testing.assert_allclose(plain, permuted, atol=1e-5)
    assert plain.shape == (1, 5)


def test_mean_pooling_remains_available_as_baseline():
    model = build_study_attention_model(4, _run().head_spec, pooling='mean')
    prediction = model({'embeddings': np.ones((1, 2, 4), np.float32),
                        'mask': np.array([[True, True]])})
    assert prediction.shape == (1, 5)


def test_smoke_test_selects_100_distinct_patients():
    splits = {
        'patient_train': list(range(1, 201)),
        'patient_valid': list(range(201, 251)),
        'patient_internal_test': list(range(251, 301)),
        'patient_test': list(range(301, 351)),
    }
    subset, patient_ids = smoke_test_splits(splits, 100)
    assert len(subset['patient_train']) == 80
    assert len(subset['patient_valid']) == 10
    assert len(subset['patient_internal_test']) == 10
    assert subset['patient_test'] == []
    assert len(patient_ids) == 100
