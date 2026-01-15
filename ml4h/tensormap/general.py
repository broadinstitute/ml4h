import os
import csv
import logging
import h5py
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict
from ml4h.TensorMap import TensorMap, Interpretation
from ml4h.normalizer import Standardize


def tensor_path(path_prefix: str, name: str) -> str:
    """
    In the future, TMAPs should be generated using this same function
    """
    return f'/{path_prefix}/{name}/'


def tensor_from_hd5(tm: TensorMap, hd5: h5py.File, dependents: Dict = {}) -> np.ndarray:
    return np.array(hd5[tm.name])

def named_tensor_from_hd5(name):
    def _tensor_from_hd5(tm: TensorMap, hd5: h5py.File, dependents: Dict = {}) -> np.ndarray:
        return np.array(hd5[name])
    return _tensor_from_hd5


def all_dates(hd5: h5py.File, path_prefix: str, name: str) -> List[str]:
    """
    Gets the dates in the hd5 with path_prefix, dtype, name.
    """
    # TODO: This ideally would be implemented to not depend on the order of name,
    # date, dtype, path_prefix in the hd5s. Unfortunately, that's hard to do
    # efficiently
    return hd5[path_prefix][name]


def pass_nan(tensor):
    return tensor


def fail_nan(tensor):
    if np.isnan(tensor).any():
        raise ValueError('Tensor contains nans.')
    return tensor


def nan_to_mean(tensor, max_allowed_nan_fraction=.2):
    tensor_isnan = np.isnan(tensor)
    if np.count_nonzero(tensor_isnan) / tensor.size > max_allowed_nan_fraction:
        raise ValueError('Tensor contains too many nans.')
    tensor[tensor_isnan] = np.nanmean(tensor)
    return tensor


def get_tensor_at_first_date(
    hd5: h5py.File,
    path_prefix: str,
    name: str,
    handle_nan=fail_nan,
):
    """
    Gets the numpy array at the first date of path_prefix, dtype, name.
    """
    dates = all_dates(hd5, path_prefix, name)
    if not dates:
        raise ValueError(f'No {name} values values available.')
    tensor = np.array(
        hd5[f'{tensor_path(path_prefix=path_prefix, name=name)}{min(dates)}/'],
        dtype=np.float32,
    )
    tensor = handle_nan(tensor)
    return tensor

def get_tensor_at_last_date(
    hd5: h5py.File,
    path_prefix: str,
    name: str,
    handle_nan=fail_nan,
):
    """
    Gets the numpy array at the last date of path_prefix, dtype, name.
    """
    dates = all_dates(hd5, path_prefix, name)
    if not dates:
        raise ValueError(f'No {name} values values available.')
    tensor = np.array(
        hd5[f'{tensor_path(path_prefix=path_prefix, name=name)}{max(dates)}/'],
        dtype=np.float32,
    )
    tensor = handle_nan(tensor)
    return tensor


def pad_or_crop_array_to_shape(new_shape: Tuple, original: np.ndarray):
    if new_shape == original.shape:
        return original
    result = np.zeros(new_shape)
    slices = tuple(
        slice(min(original.shape[i], new_shape[i]))
        for i in range(len(original.shape))
    )

    # Allow expanding one dimension eg (256, 256) can become (256, 256, 1)
    if len(new_shape) - len(original.shape) == 1:
        padded = result[..., 0]
    else:
        padded = result

    padded[slices] = original[slices]
    return result


def normalized_first_date(tm: TensorMap, hd5: h5py.File, dependents=None):
    tensor = get_tensor_at_first_date(hd5, tm.path_prefix, tm.name)
    if tm.axes() > 1:
        return pad_or_crop_array_to_shape(tm.shape, tensor)
    else:
        return tensor


def build_tensor_from_file(
    file_name: str,
    target_column: str,
    normalization: bool = False,
):
    """
    Build a tensor_from_file function from a column in a file.
    Only works for continuous values.
    When normalization is True values will be normalized according to the mean and std of all of the values in the column.
    """
    error = None
    try:
        with open(file_name, 'r') as f:
            ext = file_name.split('.')[1]
            delimiter = ',' if ext == 'csv' else '\t'
            reader = csv.reader(f, delimiter=delimiter)
            header = next(reader)
            index = header.index(target_column)
            table = {}
            for row in reader:
                try:
                    table[row[0]] = np.array([float(row[index])])
                except ValueError:
                    logging.debug(f'ValueError parsing: {row[index]}')
            if normalization:
                value_array = np.array(
                    [sub_array[0] for sub_array in table.values()],
                )
                mean = value_array.mean()
                std = value_array.std()
                logging.info(
                    f'Normalizing TensorMap from file {file_name}, column {target_column} with mean: '
                    f'{mean:.2f}, std: {std:.2f}', )
    except FileNotFoundError as e:
        error = e

    def tensor_from_file(tm: TensorMap, hd5: h5py.File, dependents=None):
        if error:
            raise error
        if normalization:
            tm.normalization = Standardize(mean=mean, std=std)
        try:
            return table[
                os.path.basename(hd5.filename).replace('.hd5', '')
            ].copy()
        except KeyError:
            raise KeyError(f'Sample id not in file {file_name}.')

    return tensor_from_file


def build_categorical_tensor_from_file(
    file_name: str,
    target_column: str,
):
    """
    Build a tensor_from_file function from a column in a file as categorical classifier.
    """
    error = None
    try:
        ext = file_name.split('.')[1]
        delimiter = ',' if ext == 'csv' else '\t'
        df = pd.read_csv(file_name, delimiter=delimiter)
        table = dict(zip(df[df.columns[0]].tolist(), df[target_column].tolist()))
        logging.info(f'Categorical table from column {target_column} counts:\n{df[target_column].value_counts()}')
    except FileNotFoundError as e:
        error = e

    def tensor_from_file(tm: TensorMap, hd5: h5py.File, dependents=None):
        if error:
            raise error
        try:
            tensor = np.zeros(tm.shape, dtype=np.float32)
            val = table[int(os.path.basename(hd5.filename).replace('.hd5', ''))]
            tensor[tm.channel_map[f'val_{val}']] = 1.0
            return tensor
        except KeyError as e:
            raise KeyError(f'Sample id not in file {file_name}, Error: {e}.')

    return tensor_from_file
