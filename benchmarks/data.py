import os
import time
from typing import Tuple, Dict, Optional, List
from ml4h.defines import TENSOR_EXT
from ml4h.TensorMap import TensorMap, Interpretation
import h5py
import numpy as np

from ml4h.defines import StorageType


Shape = Tuple[Optional[int], ...]


SYNTHETIC_DATA_PATH = os.path.join(os.path.dirname(__file__), 'synthetic_data')


def random_concrete_shape(shape: Shape) -> Tuple[int, ...]:
    return tuple(x if x is not None else 1 + np.random.randint(10) for x in shape)


def build_example(shape: Shape, storage_type: StorageType) -> np.ndarray:
    shape = random_concrete_shape(shape)
    if storage_type == StorageType.CONTINUOUS:
        return np.random.randn(shape)
    if storage_type == StorageType.STRING:
        letters = list('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
        return np.random.choice(letters, shape)
    else:
        raise NotImplementedError(f'Random generation not implemented for {storage_type}')


def build_row(
        data_descriptions: List[Tuple[str, Shape, StorageType]]
) -> Dict[str, np.ndarray]:
    return {
        name: build_example(shape, storage_type)
        for name, shape, storage_type in data_descriptions
    }


def _hd5_path_to_int(path: str) -> int:
    return int(os.path.basename(path).replace(TENSOR_EXT, ''))


def _int_to_hd5_path(i: int) -> str:
    return os.path.join(SYNTHETIC_DATA_PATH, f'{i}{TENSOR_EXT}')


def get_hd5_paths(overwrite: bool, num_hd5s: int) -> List[str]:
    if overwrite:
        to_write = list(range(num_hd5s))
    else:
        to_write = [i for i in range(num_hd5s) if not os.path.exists(_int_to_hd5_path(i))]
    return [_int_to_hd5_path(i) for i in to_write]


def write_in_hd5_ukbb(
    name: str, storage_type: StorageType, value: np.ndarray, hd5: h5py.File,
):
    """Replicates storage behavior in tensor_writer_ukbb"""
    if storage_type == StorageType.STRING:
        hd5.create_dataset(name, data=value, dtype=h5py.special_dtype(vlen=str))
    elif storage_type == StorageType.CONTINUOUS:
        hd5.create_dataset(name, data=value, compression='gzip')
    else:
        raise NotImplementedError(f'{storage_type} cannot be automatically written yet')


def build_hd5s_ukbb(
        data_descriptions: List[Tuple[str, Shape, StorageType]], num_hd5s: int,
        overwrite: bool = True,
) -> List[str]:
    paths = get_hd5_paths(overwrite, num_hd5s)
    start_time = time.time()
    print(f'Beginning to write {len(paths)} hd5s')
    for i, path in enumerate(paths):
        with h5py.File(path, 'w') as hd5:
            for name, shape, storage_type in data_descriptions:
                data = build_example(shape, storage_type)
                write_in_hd5_ukbb(name, storage_type, data, hd5)
        print(f'{(i + 1) / len(paths):.1%} done', end='\r')
    print()
    delta = time.time() - start_time
    print(f'Wrote {len(paths)} hd5s in {delta:.1f} seconds at {delta / len(paths):.1f} paths/s')
    return paths


STORAGE_TYPE_TO_INTERPRETATION = {
    StorageType.CONTINUOUS: Interpretation.CONTINUOUS,
    StorageType.STRING: Interpretation.LANGUAGE,
}


def build_tensor_maps(
    data_descriptions: List[Tuple[str, Shape, StorageType]],
) -> List[TensorMap]:
    tmaps = []
    for name, shape, storage_type in data_descriptions:
        tmaps.append(TensorMap(
            name,
            interpretation=STORAGE_TYPE_TO_INTERPRETATION[storage_type],
            shape=shape,
        ))
    return tmaps
