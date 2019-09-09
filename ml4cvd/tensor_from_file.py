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
    visited = set()
    def filter_path(path):
        if source in path and str(dtype) in path and name in path and path not in visited:
            return path
        return None

    while True:
        found = hd5.visit(filter_path)
        if found:
            visited.add(found)
            yield found
        else:
            break


def get_tensor_at_first_date(hd5: h5py.File, source: str, dtype: DataSetType, name: str):
    """
    Gets the value of the first date.
    """
    first_date_path = min(all_dates(hd5, source, dtype, name))  # Only difference in paths is date and dates sort
    return np.array(hd5[first_date_path])


def float_array_from_file_first_date(tm: TensorMap, hd5: h5py.File, dependents=None):
    return tm.zero_mean_std1(get_tensor_at_first_date(hd5, tm.group, DataSetType.FLOAT_ARRAY, tm.name).reshape(tm.shape))


def float_array_zero_pad_from_file_first_date(tm: TensorMap, hd5: h5py.File, dependents=None):
    """
    Assumes first dimension is the short dimension
    """
    padded = np.zeros(tm.shape)
    original = get_tensor_at_first_date(hd5, tm.group, DataSetType.FLOAT_ARRAY, tm.name)
    original = tm.zero_mean_std1(original)

    # The reshape accounts for either array having an extra dimension at the end. e.g. (5, 1) vs. (5,)
    padded[:original.shape[0]] = original.reshape((original.shape[0],) + tm.shape[1:])
    return padded


def first_date_bike_recovery(tm: TensorMap, hd5: h5py.File, dependents=None):
    recovery_len = get_tensor_at_first_date(hd5, 'ecg_bike', DataSetType.CONTINUOUS, 'rest_duration')
    if recovery_len != 60:
        raise ValueError(f'No recovery phase in {hd5}')
    original = get_tensor_at_first_date(hd5, tm.group, DataSetType.FLOAT_ARRAY, tm.name)
    flat = original.mean(axis=1)  # all leads are basically the same
    recovery = flat[-tm.shape[0]:]
    return tm.zero_mean_std1(recovery).reshape(tm.shape)


def continuous_from_file_first_date(tm: TensorMap, hd5: h5py.File, dependents=None):
    return tm.normalize(get_tensor_at_first_date(hd5, tm.group, DataSetType.CONTINUOUS, tm.name))
