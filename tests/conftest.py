# Imports: standard library
import os
import sys
from typing import Dict, List, Tuple
from unittest import mock as mock
from itertools import product

# Imports: third party
import h5py
import numpy as np
import pytest

# Imports: first party
from ml4cvd.defines import TENSOR_EXT
from ml4cvd.arguments import TMAPS, parse_args
from ml4cvd.TensorMap import TensorMap, Interpretation


def pytest_configure():
    pytest.N_TENSORS = 50
    pytest.CONTINUOUS_TMAPS = [
        TensorMap(
            f"{n}d_cont",
            shape=tuple(range(2, n + 2)),
            interpretation=Interpretation.CONTINUOUS,
        )
        for n in range(1, 6)
    ]
    pytest.CATEGORICAL_TMAPS = [
        TensorMap(
            f"{n}d_cat",
            shape=tuple(range(2, n + 2)),
            interpretation=Interpretation.CATEGORICAL,
            channel_map={f"c_{i}": i for i in range(n + 1)},
        )
        for n in range(1, 6)
    ]
    pytest.TMAPS_UP_TO_4D = pytest.CONTINUOUS_TMAPS[:-1] + pytest.CATEGORICAL_TMAPS[:-1]
    pytest.TMAPS_5D = pytest.CONTINUOUS_TMAPS[-1:] + pytest.CATEGORICAL_TMAPS[-1:]
    pytest.MULTIMODAL_UP_TO_4D = [
        list(x)
        for x in product(pytest.CONTINUOUS_TMAPS[:-1], pytest.CATEGORICAL_TMAPS[:-1])
    ]
    pytest.SEGMENT_IN = TensorMap(
        f"2d_for_segment_in",
        shape=(32, 32, 1),
        interpretation=Interpretation.CONTINUOUS,
        metrics=["mse"],
    )
    pytest.SEGMENT_OUT = TensorMap(
        f"2d_for_segment_out",
        shape=(32, 32, 2),
        interpretation=Interpretation.CATEGORICAL,
        channel_map={"yes": 0, "no": 1},
    )
    pytest.MOCK_TMAPS = {
        tmap.name: tmap for tmap in pytest.CONTINUOUS_TMAPS + pytest.CATEGORICAL_TMAPS
    }
    pytest.PARENT_TMAPS = [
        TensorMap(
            f"parent_test_{i}", shape=(1,), interpretation=Interpretation.CONTINUOUS,
        )
        for i in range(3)
    ]
    for i in range(len(pytest.PARENT_TMAPS)):
        pytest.PARENT_TMAPS[i].parents = pytest.PARENT_TMAPS[:i]
    pytest.CYCLE_PARENTS = [
        TensorMap(
            f"parent_test_cycle_{i}",
            shape=(1,),
            interpretation=Interpretation.CONTINUOUS,
        )
        for i in range(3)
    ]
    for i in range(len(pytest.CYCLE_PARENTS)):
        pytest.CYCLE_PARENTS[i].parents = [
            pytest.CYCLE_PARENTS[i - 1],
        ]  # 0th tmap will be child of last


pytest_configure()


class Utils:
    @staticmethod
    def build_hdf5s(
        path: str, tensor_maps: List[TensorMap], n=5,
    ) -> Dict[Tuple[str, TensorMap], np.ndarray]:
        """
        Builds hdf5s at path given TensorMaps. Only works for Continuous and Categorical TensorMaps.
        """
        out = {}
        for i in range(n):
            hd5_path = os.path.join(path, f"{i}{TENSOR_EXT}")
            with h5py.File(hd5_path, "w") as hd5:
                for tm in tensor_maps:
                    if tm.is_continuous():
                        value = np.full(tm.shape, fill_value=i, dtype=np.float32)
                    elif tm.is_categorical():
                        value = np.zeros(tm.shape, dtype=np.float32)
                        value[..., i % tm.shape[-1]] = 1
                    else:
                        raise NotImplementedError(
                            "Cannot automatically build hdf5 from interpretation"
                            f' "{tm.interpretation}"',
                        )
                    hd5.create_dataset(tm.hd5_key_guess(), data=value)
                    out[(hd5_path, tm)] = value
        return out


@pytest.fixture(scope="session")
def utils():
    return Utils


@pytest.fixture(scope="class")
@mock.patch.dict(TMAPS, pytest.MOCK_TMAPS)
def default_arguments(tmpdir_factory, utils):
    temp_dir = tmpdir_factory.mktemp("data")
    utils.build_hdf5s(temp_dir, pytest.MOCK_TMAPS.values(), n=pytest.N_TENSORS)
    hdf5_dir = str(temp_dir)
    inp_key = "3d_cont"
    out_key = "1d_cat"
    sys.argv = [
        "",
        "--output_folder",
        hdf5_dir,
        "--input_tensors",
        inp_key,
        "--output_tensors",
        out_key,
        "--tensors",
        hdf5_dir,
        "--pool_x",
        "1",
        "--pool_y",
        "1",
        "--pool_z",
        "1",
        "--training_steps",
        "2",
        "--test_steps",
        "3",
        "--validation_steps",
        "2",
        "--epochs",
        "2",
        "--num_workers",
        "0",
        "--batch_size",
        "2",
    ]
    args = parse_args()
    return args
