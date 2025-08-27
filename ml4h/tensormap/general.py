import os
import csv
import logging
import h5py
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict
from ml4h.TensorMap import TensorMap, Interpretation
from ml4h.normalizer import Standardize, StandardizeIgnoringWeights


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
    weight_column: str = None,
    filter_column: str = None,
    filter_value: int = None,
    replacement_filter_value: int = None,
):
    """
    Build a tensor_from_file function from a column in a file.
    Only works for continuous values.
    When normalization is True values will be normalized according to the mean and std of all of the values in the column.
    If weight_column is given, will return a 2D tensor that includes weight in the second column, to be used by the
    weighted_mse loss
    If filter_column and filter_value are given, will drop all values where filter_column != filter_value. You can
    optionally provide a replacement_filter_value to replace target_column entries that would be filtered out with
    target_column entries from rows where filter_column = replacment_filter_value.
    """
    error = None
    try:
        ext = file_name.split('.')[1]
        delimiter = ',' if ext == 'csv' else '\t'
        df = pd.read_csv(file_name, delimiter=delimiter, dtype={0:str})
        id_column = df.columns[0]

        if (filter_column is not None) and (filter_value is not None):

            # Remove all null values from e.g., instance 2
            condition_to_drop = (df[filter_column] == filter_value) & (df[target_column].isna())
            df = df[~condition_to_drop]

            # Find IDs that have the source instance (e.g., instance 0) but not the target instance (e.g., instance 2),
            # and use those instead
            source_ids = set(df.loc[df[filter_column] == replacement_filter_value, id_column].unique())
            target_ids = set(df.loc[df[filter_column] == filter_value, id_column].unique())
            ids_to_replace = list(source_ids - target_ids)
            if ids_to_replace:
                mask_to_relabel = (df[id_column].isin(ids_to_replace)) & (df[filter_column] == replacement_filter_value)
                df.loc[mask_to_relabel, filter_column] = filter_value

            # now do the filtering
            df = df[df[filter_column] == filter_value]

        value_cols = [target_column]
        if weight_column is not None:
            value_cols.append(weight_column)
        for c in value_cols:
            df[c] = df[c].astype(float)
        df.dropna(subset=value_cols, inplace=True)

        # Set the ID column as the index and convert the remaining value columns to a dictionary of numpy arrays
        table = df.set_index(id_column)[value_cols].apply(np.array, axis=1).to_dict()

        if normalization:
            mean = df[target_column].mean()
            std = df[target_column].std(ddof=0)
            logging.info(
                f'Normalizing TensorMap from file {file_name}, column {target_column} with mean: '
                f'{mean:.2f}, std: {std:.2f}', )
    except FileNotFoundError as e:
        error = e

    def tensor_from_file(tm: TensorMap, hd5: h5py.File, dependents=None):
        if error:
            raise error
        if normalization:
            if weight_column is None:
                tm.normalization = Standardize(mean=mean, std=std)
            else:
                tm.normalization = StandardizeIgnoringWeights(mean=mean, std=std)
        try:
            return table[
                os.path.basename(hd5.filename).replace('.hd5', '')
            ].copy()
        except KeyError as e:
            raise KeyError(f'Sample id not in file {file_name}, Error: {e}.')

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
            tensor[tm.channel_map[val]] = 1.0
            return tensor
        except KeyError as e:
            raise KeyError(f'Sample id not in file {file_name}, Error: {e}.')

    return tensor_from_file
