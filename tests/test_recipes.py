# Imports: standard library
import os

# Imports: third party
import numpy as np
import pandas as pd
import pytest

# Imports: first party
from ml4cvd.plots import _find_negative_label_index
from ml4cvd.recipes import infer_multimodal_multitask, train_multimodal_multitask
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

    def test_infer(self, default_arguments):
        infer_multimodal_multitask(default_arguments)
        path = os.path.join(
            default_arguments.output_folder,
            default_arguments.id,
            f"predictions_test.csv",
        )
        predictions = pd.read_csv(path)
        assert (
            len(set(predictions["sample_id"]))
            == default_arguments.test_steps * default_arguments.batch_size
        )

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
        default_arguments.explore_export_fpath = True
        explore(default_arguments)

        csv_path = os.path.join(
            default_arguments.output_folder, default_arguments.id, "tensors_union.csv",
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
