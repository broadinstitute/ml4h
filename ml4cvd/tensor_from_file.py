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

    # TODO: pretty inefficient, searches through every path in hd5
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


def pad_array_to_shape(tm: TensorMap, original: np.ndarray):
    padded = np.zeros(tm.shape)
    padded[:original.shape[0]] = original.reshape((original.shape[0],) + tm.shape[1:])
    return padded


def normalized_first_date(tm: TensorMap, hd5: h5py.File, dependents=None):
    tensor = get_tensor_at_first_date(hd5, tm.group, tm.dtype, tm.name)
    if tm.dtype == DataSetType.CONTINUOUS:
        return tm.normalize(tensor)
    if tm.dtype == DataSetType.FLOAT_ARRAY:
        tensor = tm.zero_mean_std1(tensor)
        return pad_array_to_shape(tm, tensor)
    raise ValueError(f'normalize_first_date not implemented for {tm.dtype}')


# BIKE ECG
def _check_phase_full_len(hd5: h5py.File, phase: str):
    phase_len = get_tensor_at_first_date(hd5, 'ecg_bike', DataSetType.CONTINUOUS, f'{phase}_duration')
    valid = True
    if phase == 'pretest':
        valid &= phase_len == 15
    elif phase == 'exercise':
        valid &= phase_len == 300
    elif phase == 'rest':
        valid &= phase_len == 60
    else:
        raise ValueError(f'Phase {phase} is not a valid phase.')
    if not valid:
        raise ValueError(f'{phase} phase is not full in {hd5}')


def first_date_bike_recovery(tm: TensorMap, hd5: h5py.File, dependents=None):
    _check_phase_full_len(hd5, 'rest')
    original = get_tensor_at_first_date(hd5, tm.group, DataSetType.FLOAT_ARRAY, tm.name)
    flat = original.mean(axis=1)  # all leads are basically the same
    recovery = flat[-tm.shape[0]:]
    return tm.zero_mean_std1(recovery).reshape(tm.shape)


def first_date_bike_pretest(tm: TensorMap, hd5: h5py.File, dependents=None):
    _check_phase_full_len(hd5, 'pretest')
    original = get_tensor_at_first_date(hd5, tm.group, DataSetType.FLOAT_ARRAY, tm.name)
    flat = original.mean(axis=1)  # all leads are basically the same
    recovery = flat[:tm.shape[0]]
    return tm.zero_mean_std1(recovery).reshape(tm.shape)


def first_date_hrr(tm: TensorMap, hd5: h5py.File, dependents=None):
    _check_phase_full_len(hd5, 'rest')
    last_hr = get_tensor_at_first_date(hd5, 'ecg_bike', DataSetType.FLOAT_ARRAY, 'trend_heartrate')[-1]
    max_hr = get_tensor_at_first_date(hd5, 'ecg_bike', DataSetType.CONTINUOUS, 'max_hr')
    return tm.normalize(last_hr - max_hr)
