import numpy as np
import datetime
from ml4cvd.defines import EPS

########################################
# Utility functions
########################################
def normalize_zero_mean_std1(np_tensor):
    """
    Normalize a target input tensor to zero mean and a standard deviation of one.
    """
    np_tensor -= np.mean(np_tensor)
    np_tensor /= np.std(np_tensor) + EPS
    np_tensor = np.nan_to_num(np_tensor)
    return np_tensor

# These helper subroutines should be moved out.
def _is_equal_field(field1: any, field2: any) -> bool:
    """
    We consider two fields equal if
        1) they are not functions and are equal, or
        2) either, or both, are functions and their names match
    If the fields are lists, we check for the above equality for corresponding
    elements from the list.
    """
    if isinstance(field1, list) and isinstance(field2, list):
        if len(field1) != len(field2):
            return False
        elif len(field1) == 0:
            return True

        fields1 = map(_get_name_if_function, field1)
        fields2 = map(_get_name_if_function, field2)

        return all([f1 == f2] for f1, f2 in zip(sorted(fields1), sorted(fields2)))
    else:
        return _get_name_if_function(field1) == _get_name_if_function(field2)


# These helper subroutines should be moved out.
def _get_name_if_function(field: any) -> any:
    """We assume 'field' is a function if it's 'callable()'"""
    if callable(field):
        return field.__name__
    else:
        return field

def _translate(val, cur_min, cur_max, new_min, new_max):
    val -= cur_min
    val /= (cur_max - cur_min)
    val *= (new_max - new_min)
    val += new_min
    return val


def str2date(d):
    parts = d.split('-')
    if len(parts) < 2:
        return datetime.datetime.now().date()
    return datetime.date(int(parts[0]), int(parts[1]), int(parts[2]))

def make_range_validator(minimum: float, maximum: float):
    def _range_validator(tm, tensor: np.ndarray):
        if not ((tensor > minimum).all() and (tensor < maximum).all()):
            raise ValueError(f'TensorMap {tm.name} failed range check.')
    return _range_validator


def no_nans(tm, tensor: np.ndarray):
    if np.isnan(tensor).any():
        raise ValueError(f'Skipping TensorMap {tm.name} with NaNs.')