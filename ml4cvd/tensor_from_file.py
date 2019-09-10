import h5py
import numpy as np
from ml4cvd.TensorMap import TensorMap
from typing import List
from ml4cvd.defines import DataSetType
from ml4cvd.tensor_writer_ukbb import tensor_path
import posixpath


"""
For now, all we will map `group` in TensorMap to `source` in tensor_path and `name` to `name`
"""


def all_dates(hd5: h5py.File, source: str, dtype: DataSetType, name: str) -> List[str]:
    """
    Gets all of the hd5 paths in the hd5 with source, dtype.
    """
    # TODO: This should be implemented to not depend on the order of name, date, dtype, source in the hd5s
    # Unfortunately, that's hard to do efficiently
    return [posixpath.join(source, str(dtype), date, name) 
            for date in hd5[source][str(dtype)] 
            if name in hd5[source][str(dtype)][date].keys()]


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
        valid &= phase_len == 360
    elif phase == 'rest':
        valid &= phase_len == 60
    else:
        raise ValueError(f'Phase {phase} is not a valid phase.')
    if not valid:
        raise ValueError(f'{phase} phase of length {phase_len} is not full in {hd5}')


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


def narrow_check(hd5, max_hr, max_pred):
    _check_phase_full_len(hd5, 'exercise')
    low, hi = 0.6551724137931034, 0.6927710843373494
    ratio = max_hr / max_pred
    if low < ratio < hi:
        return
    raise ValueError(f'Max hr / max pred hr {ratio} not in {low, hi}')
    

def narrow_hrr(tm: TensorMap, hd5: h5py.File, dependents=None):
    last_hr = get_tensor_at_first_date(hd5, 'ecg_bike', DataSetType.FLOAT_ARRAY, 'trend_heartrate')[-1]
    max_hr = get_tensor_at_first_date(hd5, 'ecg_bike', DataSetType.CONTINUOUS, 'max_hr')
    max_pred_hr = get_tensor_at_first_date(hd5, 'ecg_bike', DataSetType.CONTINUOUS, 'max_pred_hr')
    _check_phase_full_len(hd5, 'rest')
    hrr = max_hr - last_hr
    narrow_check(hd5, max_hr, max_pred_hr)
    return tm.normalize(hrr / max_pred_hr)


def narrow_max_hr(tm: TensorMap, hd5: h5py.File, dependents=None):
    max_hr = get_tensor_at_first_date(hd5, 'ecg_bike', DataSetType.CONTINUOUS, 'max_hr')
    max_pred_hr = get_tensor_at_first_date(hd5, 'ecg_bike', DataSetType.CONTINUOUS, 'max_pred_hr')
    narrow_check(hd5, max_hr, max_pred_hr)
    return tm.normalize(max_hr / max_pred_hr)
