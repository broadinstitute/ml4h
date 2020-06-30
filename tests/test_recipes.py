import os
import pytest
import pandas as pd
from typing import List

from ml4cvd.defines import SAMPLE_ID, MODEL_EXT
from ml4cvd.models import make_multimodal_multitask_model

from ml4cvd.recipes import inference_file_name, hidden_inference_file_name
from ml4cvd.recipes import train_multimodal_multitask, compare_multimodal_multitask_models
from ml4cvd.recipes import infer_multimodal_multitask, infer_hidden_layer_multimodal_multitask
from ml4cvd.recipes import compare_multimodal_scalar_task_models, _find_learning_rate
# Imports with test in their name
from ml4cvd.recipes import test_multimodal_multitask as tst_multimodal_multitask
from ml4cvd.recipes import test_multimodal_scalar_tasks as tst_multimodal_scalar_tasks


def init_models(args, num_models: int) -> List[str]:
    paths = []
    for i in range(num_models):
        m = make_multimodal_multitask_model(**args.__dict__)
        path = os.path.join(args.output_folder, f'test_model_{i}{MODEL_EXT}')
        m.save(path)
        paths.append(path)
    return paths


class TestRecipes:
    """Smoke tests"""

    def test_train(self, default_arguments):
        train_multimodal_multitask(default_arguments)

    def test_test(self, default_arguments):
        default_arguments.model_file = init_models(default_arguments, 1)[0]
        tst_multimodal_multitask(default_arguments)

    def test_test_scalar(self, default_arguments):
        default_arguments.model_file = init_models(default_arguments, 1)[0]
        tst_multimodal_scalar_tasks(default_arguments)

    @pytest.mark.parametrize('num_workers', [0, 3])
    def test_infer(self, default_arguments, num_workers):
        default_arguments.num_workers = num_workers
        default_arguments.model_files = init_models(default_arguments, 2)
        default_arguments.model_ids = ['m0', 'm1']
        infer_multimodal_multitask(default_arguments)
        tsv = inference_file_name(default_arguments.output_folder, default_arguments.id)
        inferred = pd.read_csv(tsv, sep='\t')
        assert len(set(inferred[SAMPLE_ID])) == pytest.N_TENSORS

    def test_infer_genetics(self, default_arguments):
        default_arguments.model_files = init_models(default_arguments, 2)
        default_arguments.model_ids = ['m0', 'm1']
        default_arguments.tsv_style = 'genetics'
        infer_multimodal_multitask(default_arguments)
        default_arguments.tsv_style = 'standard'
        tsv = inference_file_name(default_arguments.output_folder, default_arguments.id)
        inferred = pd.read_csv(tsv, sep='\t')
        assert len(set(inferred['FID'])) == pytest.N_TENSORS

    @pytest.mark.parametrize('num_workers', [0, 3])
    def test_infer_hidden(self, default_arguments, num_workers):
        default_arguments.num_workers = num_workers
        default_arguments.model_files = init_models(default_arguments, 2)
        default_arguments.model_ids = ['m0', 'm1']
        infer_hidden_layer_multimodal_multitask(default_arguments)
        tsv = hidden_inference_file_name(default_arguments.output_folder, default_arguments.id)
        inferred = pd.read_csv(tsv, sep='\t')
        assert len(set(inferred[SAMPLE_ID])) == pytest.N_TENSORS

    def test_infer_hidden_genetics(self, default_arguments):
        default_arguments.model_files = init_models(default_arguments, 2)
        default_arguments.model_ids = ['m0', 'm1']
        default_arguments.tsv_style = 'genetics'
        infer_hidden_layer_multimodal_multitask(default_arguments)
        default_arguments.tsv_style = 'standard'
        tsv = hidden_inference_file_name(default_arguments.output_folder, default_arguments.id)
        inferred = pd.read_csv(tsv, sep='\t')
        assert len(set(inferred['FID'])) == pytest.N_TENSORS

    def test_find_learning_rate(self, default_arguments):
        _find_learning_rate(default_arguments)
