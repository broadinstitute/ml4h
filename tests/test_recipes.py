# Imports: standard library
import os

# Imports: third party
import numpy as np
import pandas as pd
import pytest

# Imports: first party
from ml4cvd.plots import _find_negative_label_index
from ml4cvd.recipes import inference_file_name
from ml4cvd.recipes import test_multimodal_multitask as tst_multimodal_multitask
from ml4cvd.recipes import (
    hidden_inference_file_name,
    infer_multimodal_multitask,
    train_multimodal_multitask,
)
from ml4cvd.recipes import test_multimodal_scalar_tasks as tst_multimodal_scalar_tasks
from ml4cvd.recipes import (
    compare_multimodal_multitask_models,
    compare_multimodal_scalar_task_models,
    infer_hidden_layer_multimodal_multitask,
)
from ml4cvd.TensorMap import TensorMap, Interpretation
from ml4cvd.explorations import (
    explore,
    continuous_explore_header,
    categorical_explore_header,
    tmap_requires_modification_for_explore,
)


class TestRecipes:
    def test_train(self, default_arguments):
        train_multimodal_multitask(default_arguments)

    def test_test(self, default_arguments):
        tst_multimodal_multitask(default_arguments)

    def test_test_scalar(self, default_arguments):
        tst_multimodal_scalar_tasks(default_arguments)

    # TODO either fix these or delete
    """
    def test_infer(self, default_arguments):
        infer_multimodal_multitask(default_arguments)
        tsv = inference_file_name(default_arguments.output_folder, default_arguments.id)
        inferred = pd.read_csv(tsv, sep="\t")
        assert len(set(inferred["sample_id"])) == pytest.N_TENSORS

    def test_infer_genetics(self, default_arguments):
        default_arguments.tsv_style = "genetics"
        infer_multimodal_multitask(default_arguments)
        default_arguments.tsv_style = "standard"
        tsv = inference_file_name(default_arguments.output_folder, default_arguments.id)
        inferred = pd.read_csv(tsv, sep="\t")
        assert len(set(inferred["FID"])) == pytest.N_TENSORS

    def test_infer_hidden(self, default_arguments):
        infer_hidden_layer_multimodal_multitask(default_arguments)
        tsv = hidden_inference_file_name(
            default_arguments.output_folder, default_arguments.id,
        )
        inferred = pd.read_csv(tsv, sep="\t")
        assert len(set(inferred["sample_id"])) == pytest.N_TENSORS

    def test_infer_hidden_genetics(self, default_arguments):
        default_arguments.tsv_style = "genetics"
        infer_hidden_layer_multimodal_multitask(default_arguments)
        default_arguments.tsv_style = "standard"
        tsv = hidden_inference_file_name(
            default_arguments.output_folder, default_arguments.id,
        )
        inferred = pd.read_csv(tsv, sep="\t")
        assert len(set(inferred["FID"])) == pytest.N_TENSORS
    """

    def test_explore(self, default_arguments, tmpdir_factory, utils):
        temp_dir = tmpdir_factory.mktemp("explore_tensors")
        default_arguments.tensors = str(temp_dir)
        tmaps = pytest.TMAPS_UP_TO_4D[:]
        tmaps.append(
            TensorMap(f"scalar", shape=(1,), interpretation=Interpretation.CONTINUOUS),
        )
        explore_expected = utils.build_hdf5s(temp_dir, tmaps, n=pytest.N_TENSORS)
        default_arguments.num_workers = 3
        default_arguments.tensor_maps_in = tmaps
        explore(default_arguments)
        csv_path = os.path.join(
            default_arguments.output_folder,
            default_arguments.id,
            "tensors_all_union.csv",
        )
        explore_result = pd.read_csv(csv_path)
        for row in explore_result.iterrows():
            row = row[1]
            for tm in tmaps:
                row_expected = explore_expected[(row["fpath"], tm)]
                if tmap_requires_modification_for_explore(tm):
                    actual = getattr(row, continuous_explore_header(tm))
                    assert not np.isnan(actual)
                    continue
                if tm.is_continuous():
                    actual = getattr(row, continuous_explore_header(tm))
                    assert actual == row_expected
                    continue
                if tm.is_categorical():
                    negative_label_idx = _find_negative_label_index(tm.channel_map)
                    for channel, idx in tm.channel_map.items():
                        if idx == negative_label_idx and len(tm.channel_map) == 2:
                            assert categorical_explore_header(tm, channel) not in row
                        else:
                            channel_val = getattr(
                                row, categorical_explore_header(tm, channel),
                            )
                            assert channel_val == row_expected[idx]
