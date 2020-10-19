import os
import sys
import time
import h5py
# import zarr
import numpy as np
# import dask.array as da
import tensorflow as tf
from itertools import cycle
from multiprocessing import Queue, Process, Pool
from torch.utils.data import DataLoader

from hangar import Repository
from hangar import make_tf_dataset, make_torch_dataset


from numcodecs import Blosc, blosc
from contextlib import contextmanager
# blosc.set_nthreads(2)
from typing import Tuple, Optional, List, Callable, Union, Dict

from ml4h.defines import TENSOR_EXT, StorageType
from ml4h.TensorMap import TensorMap, Interpretation


Shape = Tuple[Optional[int], ...]
DataDescription = Tuple[str, Shape, StorageType]


SYNTHETIC_DATA_PATH = os.path.join(os.path.dirname(__file__), 'synthetic_data')


def random_concrete_shape(shape: Shape) -> Tuple[int, ...]:
    return tuple(x if x is not None else 1 + np.random.randint(10) for x in shape)


def build_example(shape: Shape, storage_type: StorageType) -> np.ndarray:
    shape = random_concrete_shape(shape)
    if storage_type == StorageType.CONTINUOUS:
        return np.random.randn(*shape).astype(np.float32)
    if storage_type == StorageType.STRING:
        letters = list('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
        return np.random.choice(letters, shape)
    else:
        raise NotImplementedError(f'Random generation not implemented for {storage_type}')


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
    compression: str,
):
    """Replicates storage behavior in tensor_writer_ukbb"""
    if storage_type == StorageType.STRING:
        hd5.create_dataset(name, data=value, dtype=h5py.special_dtype(vlen=str))
    elif storage_type == StorageType.CONTINUOUS:
        hd5.create_dataset(name, data=value, compression=compression)
    else:
        raise NotImplementedError(f'{storage_type} cannot be automatically written yet')


def build_hd5s_ukbb(
        data_descriptions: List[DataDescription], num_hd5s: int,
        overwrite: bool = True, compression: str = 'gzip',
):
    paths = get_hd5_paths(overwrite, num_hd5s)
    start_time = time.time()
    print(f'Beginning to write {len(paths)} hd5s')
    for i, path in enumerate(paths):
        with h5py.File(path, 'w') as hd5:
            for name, shape, storage_type in data_descriptions:
                data = build_example(shape, storage_type)
                write_in_hd5_ukbb(name, storage_type, data, hd5, compression)
        print(f'Writing hd5s {(i + 1) / len(paths):.1%} done', end='\r')
        sys.stdout.flush()
    print()
    delta = time.time() - start_time
    print(f'Wrote {len(paths)} hd5s in {delta:.1f} seconds at {len(paths) / delta:.1f} paths/s')


# def build_zarr(
#     data_descriptions: List[DataDescription], num_samples: int, store_type=zarr.DirectoryStore,
# ):
#     start_time = time.time()
#     print(f'Beginning to write {num_samples} zarrs')
#     for i in range(num_samples):
#         store = store_type(os.path.join(SYNTHETIC_DATA_PATH, f'{i}.zarr'), 'w')
#         root = zarr.group(store=store, overwrite=True)
#         for name, shape, storage_type in data_descriptions:
#             data = build_example(shape, storage_type)
#             root.create_dataset(
#                 name, data=data, dtype=data.dtype,
#                 compressor=Blosc(cname='lz4hc', clevel=9),
#             )
#         print(f'Writing zarrs {(i + 1) / num_samples:.1%} done', end='\r')
#         sys.stdout.flush()
#     print()
#     delta = time.time() - start_time
#     print(f'Wrote {num_samples} zarrs in {delta:.1f} seconds at {num_samples / delta:.1f} paths/s')


GIANT_ZARR = os.path.join(SYNTHETIC_DATA_PATH, 'all.zarr')


def _pick_chunks(shape: Tuple[int]):
    # TODO: should be optimized
    # return tuple(min(100, s) for s in shape)
    return (None,) * len(shape)


# def build_zarr_giant_array(
#         data_descriptions: List[DataDescription], num_samples: int, store_type=zarr.DirectoryStore,
# ):
#     start_time = time.time()
#     print(f'Beginning to write {num_samples} zarrs')
#     store = store_type(GIANT_ZARR, 'w')
#     root = zarr.group(store=store, overwrite=True)
#     dsets = {
#         name: root.empty(
#             name, shape=(num_samples,) + shape,
#             compressor=Blosc(cname='lz4hc', clevel=9),
#             chunks=(1,) + _pick_chunks(shape),
#         )
#         for name, shape, _ in data_descriptions
#     }
#     for i in range(num_samples):
#         for name, shape, storage_type in data_descriptions:
#             data = build_example(shape, storage_type)
#             dsets[name][i] = data
#         print(f'Writing zarrs {(i + 1) / num_samples:.1%} done', end='\r')
#         sys.stdout.flush()
#     print()
#     delta = time.time() - start_time
#     print(f'Wrote {num_samples} zarrs in {delta:.1f} seconds at {num_samples / delta:.1f} paths/s')


class ZarrGenerator:
    def __init__(
            self, num_samples: int,
            data_descriptions: List[DataDescription],
            num_workers: int,
            batch_size: int,
            giant: bool,
            zarr_store,
    ):
        paths = list(range(num_samples))
        self.split_paths = np.array_split(paths, num_workers)
        self.q = None
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.data_descriptions = data_descriptions
        self.workers: List[Process] = []
        self._started = False
        self.giant = giant
        self.store = zarr_store

    def _init_workers(self):
        self.q = Queue(self.batch_size)
        self._started = True
        for i, paths in enumerate(self.split_paths):
            worker_instance = _ZarrWorker(
                paths,
                self.data_descriptions,
                self.q,
                self.batch_size,
                self.store,
            )
            process = Process(
                target=worker_instance.big_array_worker if self.giant else worker_instance.worker,
                args=(),
            )
            process.start()
            self.workers.append(process)

    def __next__(self):
        if not self._started:
            self._init_workers()
        return self.q.get(64)

    def kill(self):
        if self._started:
            for worker in self.workers:
                worker.terminate()


class _ZarrWorker:

    def __init__(
            self, paths: List[str],
            data_descriptions: List[DataDescription],
            q: Queue,
            batch_size: int,
            store,
    ):
        self.paths = paths
        self.data_descriptions = data_descriptions
        self.q = q
        self.batch_size = batch_size
        self.idx = 0
        self.batch = None
        self.empty_batch()
        self.store = store

    def empty_batch(self):
        self.batch = {
            name: np.zeros((self.batch_size,) + shape)
            for name, shape, _ in self.data_descriptions
        }

    def worker(self):
        for path in cycle(self.paths):
            path = os.path.join(SYNTHETIC_DATA_PATH, f'{path}.zarr')
            with zarr.open(self.store(path), 'r') as z:
                for name, _, _ in self.data_descriptions:
                    self.batch[name][self.idx] = z[name]
            self.idx += 1
            if self.idx == self.batch_size:
                self.q.put(self.batch)
                self.idx = 0
                self.empty_batch()

    def big_array_worker(self):
        with zarr.open(self.store(GIANT_ZARR), 'r') as z:
            for path in cycle(self.paths):
                for name, _, _ in self.data_descriptions:
                    self.batch[name][self.idx] = z[name][path]
                self.idx += 1
                if self.idx == self.batch_size:
                    self.q.put(self.batch)
                    self.idx = 0
                    self.empty_batch()


def tf_data_zarr(
        num_samples: int,
        data_descriptions: List[DataDescription],
        num_workers: int,
        batch_size: int,
        zarr_store,
):
    return tf.data.Dataset.range(num_samples).shuffle(num_samples).repeat().interleave(
        lambda idx: tf.data.Dataset.from_generator(
            _ZarrTFDataGenerator(zarr_store, data_descriptions),
            args=(idx,), output_types=tf_data_types(data_descriptions),
            output_shapes=tf_output_shapes(data_descriptions),
        ),
        block_length=1, cycle_length=num_workers,
        num_parallel_calls=num_workers,
    ).batch(batch_size).as_numpy_iterator()


def tf_output_shapes(data_descriptions: List[DataDescription]):
    return {name: shape for name, shape, _ in data_descriptions}


def tf_data_types(data_descriptions: List[DataDescription]):
    return {name: tf.float32 for name, _, _ in data_descriptions}


class _ZarrTFDataGenerator:

    def __init__(self, store, data_descriptions: List[DataDescription]):
        self.store = store
        self.data_descriptions = data_descriptions

    def __call__(self, path: int):
        path = os.path.join(SYNTHETIC_DATA_PATH, f'{path}.zarr')
        with zarr.open(self.store(path), 'r') as z:
            batch = {}
            for name, _, _ in self.data_descriptions:
                batch[name] = z[name]
            yield batch


class ZarrSequence(tf.keras.utils.Sequence):

    def __init__(
        self,
        num_samples: int,
        data_descriptions: List[DataDescription],
        batch_size: int,
        zarr_store,
    ):
        self.epoch_len = int(np.ceil(num_samples / batch_size))
        self.batches = np.array_split(
            list(range(num_samples)), self.epoch_len
        )
        self.store = zarr_store
        self.data_descriptions = data_descriptions
        self.file = zarr.open(self.store(GIANT_ZARR), 'r')

    def __len__(self):
        return self.epoch_len

    def __getitem__(self, item: int):
        idx = self.batches[item]
        out = {}
        for name, _, _ in self.data_descriptions:
            out[name] = self.file[name].oindex[idx, :]
        return out

    def close(self):
        self.file.close()


class DaskZarrGenerator:

    def __init__(
        self, num_samples: int,
        data_descriptions: List[DataDescription],
        num_workers: int,
        batch_size: int,
        zarr_store,
    ):
        self.paths = list(range(num_samples))
        self.num_samples = num_samples
        self.batch_size = batch_size
        self.epoch_idx = 0
        self.max_epoch_index = np.ceil(self.num_samples / self.batch_size)
        self.num_workers = num_workers
        self.data_descriptions = data_descriptions
        self.store = zarr_store
        self.zarr_arrays = {
            da.from_zarr(f'{SYNTHETIC_DATA_PATH}')
        }

    def _build_batches(self):
        np.random.shuffle(self.paths)
        self.batches = np.array_split(self.paths, self.max_epoch_index)

    def __next__(self):
        if self.epoch_idx == self.max_epoch_index:
            self._build_batches()
        batch_ids = self.batches[self.epoch_idx]


TFDtype = Union[List[tf.dtypes.DType], tf.dtypes.DType, Dict[str, tf.dtypes.DType]]


class _ParallelPyFuncWrapper:
    def __init__(self, map_function, num_workers: int):
        self.map_function = map_function
        self.num_workers = num_workers
        self.pool = Pool(num_workers)

    def send_to_map_pool(self, x):
        """
        Sends the tensor element to the pool for processing.

        :param x: The element to be processed by the pool.
        :return: The output of the map function on the element.
        """
        result = self.pool.apply_async(self.map_function, (x,))
        mapped_element = result.get()
        return mapped_element

    def map_to_dataset(
            self, dataset: tf.data.Dataset, output_types: TFDtype
    ):
        """
        Maps the map function to the passed dataset.

        :param dataset: The dataset to apply the map function to.
        :param output_types: The TensorFlow output types of the function to convert to.
        :return: The mapped dataset.
        """
        def map_py_function(*args):
            """A py_function wrapper for the map function."""
            return tf.py_function(self.send_to_map_pool, args, output_types)

        return dataset.map(map_py_function, self.num_workers)


def map_py_function_to_dataset(
        dataset: tf.data.Dataset, map_function: Callable, number_of_parallel_calls: int,
        output_types: TFDtype,
) -> tf.data.Dataset:
    """
    A one line wrapper to allow mapping a parallel py function to a dataset.

    :param dataset: The dataset whose elements the mapping function will be applied to.
    :param map_function: The function to map to the dataset.
    :param number_of_parallel_calls: The number of parallel calls of the mapping function.
    :param output_types: The TensorFlow output types of the function to convert to.
    :return: The mapped dataset.
    """
    py_mapper = _ParallelPyFuncWrapper(map_function=map_function, num_workers=number_of_parallel_calls)
    mapped_dataset = py_mapper.map_to_dataset(dataset=dataset, output_types=output_types)
    return mapped_dataset


class _ZarrMap:

    def __init__(
            self,
            num_samples: int,
            data_descriptions: List[DataDescription],
            zarr_store,
    ):
        self.store = zarr_store
        self.data_descriptions = data_descriptions
        self.file = zarr.open(self.store(GIANT_ZARR), 'r')
        self.num_samples = num_samples

    def get_idx(self, item: int):
        return [self.file[name][int(item)] for name, _, _ in self.data_descriptions]

    def to_dict(self, *args):
        return {name: x for (name, _, _), x in zip(self.data_descriptions, args)}

    def close(self):
        pass
        # self.file.close()

    def build_dataset(self, num_workers: int, batch_size: int):
        dataset = tf.data.Dataset.range(self.num_samples).shuffle(self.num_samples).repeat()
        mapped_dataset = map_py_function_to_dataset(
            dataset, self.get_idx, num_workers, list(tf_data_types(self.data_descriptions).values()),
        ).map(self.to_dict)
        return mapped_dataset.batch(batch_size).as_numpy_iterator()


def build_hangar_repo(num_samples: int, data_descriptions: List[DataDescription]):
    repo = Repository(SYNTHETIC_DATA_PATH)
    repo.init('synth', 'ml4h@broadinistitute.org', remove_old=True)
    co = repo.checkout(write=True)
    cols = {
        name: co.add_ndarray_column(name, shape=shape, dtype=np.float32)
        for name, shape, _ in data_descriptions
    }
    for i in range(num_samples):
        for name, shape, _ in data_descriptions:
            cols[name][i] = build_example(shape, StorageType.CONTINUOUS)
        print(f'Writing hangar rows {(i + 1) / num_samples:.1%} done', end='\r')
        sys.stdout.flush()
    print()
    co.commit('added data')
    co.close()


@contextmanager
def make_hangar_tf_dset(data_descriptions: List[DataDescription], batch_size: int):
    repo = Repository(SYNTHETIC_DATA_PATH)
    co = repo.checkout(write=False)
    dset = make_tf_dataset(columns=[co.columns[name] for name, _, _ in data_descriptions])
    yield dset.batch(batch_size).as_numpy_iterator()
    co.close()


@contextmanager
def make_hangar_torch_dset(data_descriptions: List[DataDescription], batch_size: int, num_workers: int):
    repo = Repository(SYNTHETIC_DATA_PATH)
    co = repo.checkout(write=False)
    dset = make_torch_dataset(columns=[co.columns[name] for name, _, _ in data_descriptions])
    yield iter(DataLoader(dset, batch_size=batch_size, num_workers=num_workers))
    co.close()


STORAGE_TYPE_TO_INTERPRETATION = {
    StorageType.CONTINUOUS: Interpretation.CONTINUOUS,
    StorageType.STRING: Interpretation.LANGUAGE,
}


def build_tensor_maps(
    data_descriptions: List[DataDescription],
) -> List[TensorMap]:
    tmaps = []
    for name, shape, storage_type in data_descriptions:
        tmaps.append(
            TensorMap(
                name,
                interpretation=STORAGE_TYPE_TO_INTERPRETATION[storage_type],
                shape=shape,
            ),
        )
    return tmaps