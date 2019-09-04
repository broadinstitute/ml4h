import h5py
import numpy as np
from ml4cvd.TensorMap import TensorMap

from ml4cvd.defines import DataSetType


"""
For now, all we will map `group` in TensorMap to `source` in tensor_path and `name` to `name`
"""


def all_dates(hd5: h5py.File, source: str, dtype: DataSetType, name: str) -> str:
    """
    Gets all of the paths in the hd5 with source, dtype, name
    """
    def filter_path(path):
        if source in path and str(dtype) in path and name in path:
            return path
        return None
    return hd5.visit(filter_path)


def get_tensor_at_first_date(hd5: h5py.File, source: str, dtype: DataSetType, name: str):
    """
    Gets the value of the first date.
    """
    first_date_path = min(all_dates(hd5, source, dtype, name))  # Only difference in paths is date and dates sort
    return np.ndarray(hd5[first_date_path])


def float_array_from_file_first_date(tm: TensorMap, hd5: h5py.File, dependents=None):
    return tm.zero_mean_std1(get_tensor_at_first_date(hd5, tm.group, DataSetType.FLOAT_ARRAY, tm.name).reshape(tm.shape))


def continuous_from_file_first_date(tm: TensorMap, hd5: h5py.File, dependents=None):
    return tm.normalization(get_tensor_at_first_date(hd5, tm.group, DataSetType.CONTINUOUS, tm.name))
