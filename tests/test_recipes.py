import os
import pytest
import logging
import pandas as pd
import numpy as np
import tensorflow as tf

from types import SimpleNamespace

from ml4h.recipes import inference_file_name, _hidden_file_name,save_to_google_cloud
from ml4h.recipes import train_legacy, train_multimodal_multitask
from ml4h.recipes import infer_multimodal_multitask, infer_hidden_layer_multimodal_multitask
from ml4h.recipes import infer_transformer_on_parquet_fast
from ml4h.recipes import compare_multimodal_scalar_task_models, _find_learning_rate
from ml4h.explorations import _categorical_explore_header, _should_error_detect, explore
# Imports with test in their name
from ml4h.recipes import test_multimodal_multitask as tst_multimodal_multitask
from ml4h.recipes import test_multimodal_scalar_tasks as tst_multimodal_scalar_tasks
from ml4h.test_utils import TMAPS_UP_TO_4D
from ml4h.test_utils import build_hdf5s
from ml4h.TensorMap import TensorMap, Interpretation


class TestRecipes:
    def test_infer_transformer_on_parquet_fast_outputs_unique_test_groups(self, monkeypatch, tmp_path):
        class FakeModel:
            output_names = ['target']

            def __call__(self, inputs, training=False):
                masked_num = inputs['num'] * inputs['mask'][..., None]
                preds = masked_num.sum(axis=(1, 2), keepdims=True)
                return {'target': tf.convert_to_tensor(preds, dtype=tf.float32)}

        monkeypatch.setattr('ml4h.recipes.keras.config.enable_unsafe_deserialization', lambda: None)
        monkeypatch.setattr('ml4h.recipes.keras.models.load_model', lambda _: FakeModel())

        df = pd.DataFrame(
            {
                'mrn': [101, 101, 202, 202, 303, 303],
                'visit_time': [1, 2, 1, 2, 1, 2],
                'value': [1.0, 2.0, 10.0, 20.0, 100.0, 200.0],
                'target': [0.1, 0.1, 0.2, 0.2, 0.3, 0.3],
            },
        )
        input_path = tmp_path / 'input.pq'
        df.to_parquet(input_path, index=False)

        test_csv = tmp_path / 'test.csv'
        pd.DataFrame([202, 303]).to_csv(test_csv, index=False, header=False)

        args = SimpleNamespace(
            model_file=str(tmp_path / 'fake.keras'),
            transformer_input_file=str(input_path),
            transformer_label_file=None,
            merge_columns=[],
            input_numeric_columns=['value'],
            latent_dimensions_start=0,
            latent_dimensions=0,
            input_categorical_columns=[],
            target_regression_columns=['target'],
            target_binary_columns=[],
            group_column='mrn',
            sort_column='visit_time',
            sort_column_ascend=True,
            transformer_max_size=8,
            train_csv=None,
            valid_csv=None,
            test_csv=str(test_csv),
            max_samples=1,
            batch_size=1,
            test_steps=10,
            output_folder=str(tmp_path),
            id='fast_infer_test',
        )

        infer_transformer_on_parquet_fast(args)

        output_df = pd.read_parquet(tmp_path / 'fast_infer_test' / 'predictions_fast_infer_test.pq')
        assert output_df['mrn'].tolist() == [202]
        assert output_df['n_rows'].tolist() == [2]
        assert output_df['visit_time'].tolist() == [1]
        assert output_df['target'].tolist() == [0.2]
        assert output_df['target_prediction'].tolist() == [30.0]

    def test_train(self, default_arguments):
        default_arguments.named_outputs = True
        train_multimodal_multitask(default_arguments)

    def test_train_legacy(self, default_arguments):
        default_arguments.named_outputs = True
        train_legacy(default_arguments)

    def test_test(self, default_arguments):
        tst_multimodal_multitask(default_arguments)

    def test_test_scalar(self, default_arguments):
        tst_multimodal_scalar_tasks(default_arguments)

    def test_infer(self, default_arguments):
        infer_multimodal_multitask(default_arguments)
        tsv = inference_file_name(default_arguments.output_folder, default_arguments.id)
        inferred = pd.read_csv(tsv, sep='\t')
        assert len(set(inferred['sample_id'])) == pytest.N_TENSORS

    def test_infer_genetics(self, default_arguments):
        default_arguments.tsv_style = 'genetics'
        infer_multimodal_multitask(default_arguments)
        tsv = inference_file_name(default_arguments.output_folder, default_arguments.id)
        inferred = pd.read_csv(tsv, sep='\t')
        default_arguments.tsv_style = 'standard'
        assert len(set(inferred['FID'])) == pytest.N_TENSORS

    def test_infer_hidden(self, default_arguments):
        infer_hidden_layer_multimodal_multitask(default_arguments)
        tsv = _hidden_file_name(default_arguments.output_folder, default_arguments.hidden_layer, default_arguments.id, '.tsv')
        inferred = pd.read_csv(tsv, sep='\t')
        assert len(set(inferred['sample_id'])) == pytest.N_TENSORS

    def test_find_learning_rate(self, default_arguments):
        default_arguments.named_outputs = True
        _find_learning_rate(default_arguments)

    @pytest.mark.slow
    def test_save_to_google_cloud(self,default_arguments):
        save_to_google_cloud(default_arguments)


    def test_explore(self, default_arguments, tmpdir_factory):
        temp_dir = tmpdir_factory.mktemp('explore_tensors2')
        default_arguments.tensors = str(temp_dir)
        tmaps = TMAPS_UP_TO_4D[:]
        tmaps.append(TensorMap(f'scalar', shape=(1,), interpretation=Interpretation.CONTINUOUS))
        explore_expected = build_hdf5s(temp_dir, tmaps, n=pytest.N_TENSORS, keys_are_paths=False)
        default_arguments.num_workers = 3
        default_arguments.tensor_maps_in = tmaps
        explore(default_arguments)

        csv_path = os.path.join(
            default_arguments.output_folder, default_arguments.id, 'tensors_all_union.csv',
        )
        explore_result = pd.read_csv(csv_path)
        logging.info(f'Tested explore {[c for c in explore_expected]}')
        logging.info(f'Tested explore_result {[c for c in explore_result["sample_id"]]}')
        for row in explore_result.iterrows():
            row = row[1]
            for tm in tmaps:
                row_expected = explore_expected[(row['sample_id'], tm.name)]
                if _should_error_detect(tm):
                    actual = getattr(row, tm.name)
                    assert not np.isnan(actual)
                    continue
                if tm.is_continuous():
                    actual = getattr(row, tm.name)
                    assert actual == row_expected
                    continue
                if tm.is_categorical():
                    for channel, idx in tm.channel_map.items():
                        channel_val = getattr(row, _categorical_explore_header(tm, channel))
                        assert channel_val == row_expected[idx]
