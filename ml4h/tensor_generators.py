# tensor_generators.py
#
# On-the-fly data generation of tensors for training or prediction.
#
# October 2018
# Sam Friedman
# sam@broadinstitute.org

# Python 2/3 friendly
from __future__ import print_function

# Imports
import os
import csv
import copy
import math
import h5py
import time
import logging
import traceback
import numpy as np
import pandas as pd
from collections import Counter
from multiprocessing import Process, Queue
from itertools import chain
from typing import List, Dict, Tuple, Set, Optional, Iterator, Callable, Any, Union, Type
import psutil
from functools import lru_cache
import threading
import queue
import pyarrow as pa

import tensorflow as tf
import pyarrow.dataset as ds
import pyarrow.parquet as pq
from sklearn.model_selection import train_test_split

from ml4h.defines import TENSOR_EXT, TensorGeneratorABC
from ml4h.ml4ht_integration.tensor_generator import TensorMapDataLoader
from ml4h.TensorMap import TensorMap

np.set_printoptions(threshold=np.inf)


DEFAULT_VALID_RATIO = 0.2
DEFAULT_TEST_RATIO = 0.1

TENSOR_GENERATOR_TIMEOUT = 64
TENSOR_GENERATOR_MAX_Q_SIZE = 32

# TensorGenerator batch indices
BATCH_INPUT_INDEX, BATCH_OUTPUT_INDEX, BATCH_PATHS_INDEX = 0, 1, 2

Path = str
PathIterator = Iterator[Path]
Batch = Dict[Path, np.ndarray]
BatchFunction = Callable[[Batch, Batch, bool, List[Path], 'kwargs'], Any]


class _ShufflePaths(Iterator):

    def __init__(self, paths: List[Path]):
        self.paths = paths
        np.random.shuffle(self.paths)
        self.idx = 0

    def __next__(self):
        if self.idx >= len(self.paths):
            self.idx = 0
            np.random.shuffle(self.paths)
        path = self.paths[self.idx]
        self.idx += 1
        return path


class _WeightedPaths(Iterator):

    def __init__(self, paths: List[PathIterator], weights: List[float]):
        self.paths = paths
        self.weights = weights
        if len(paths) != len(weights):
            raise ValueError('Weights must be the same length as paths.')

    def __next__(self) -> str:
        return np.random.choice(np.random.choice(self.paths, self.weights))


def pick_generator(paths, weights, mixup, siamese) -> Type[TensorGeneratorABC]:
    try:
        TensorMapDataLoader.can_apply(paths, weights, mixup, siamese)
        return TensorMapDataLoader
    except NotImplementedError as e:
        logging.warning(f"Could not use new data loader because: {repr(e)}. Defaulting to legacy TensorGenerator.")
        return TensorGenerator


class TensorGenerator(TensorGeneratorABC):
    def __init__(
        self, batch_size: int, input_maps: List[TensorMap], output_maps: List[TensorMap],
        paths: Union[List[str], List[List[str]]], num_workers: int, cache_size: float, weights: List[float] = None,
        keep_paths: bool = False, mixup_alpha: float = 0.0, name: str = 'worker', siamese: bool = False,
        augment: bool = False,
    ):
        """
        :param paths: If weights is provided, paths should be a list of path lists the same length as weights
        """
        self.augment = augment
        self.paths = sum(paths) if (len(paths) > 0 and isinstance(paths[0], list)) else paths
        self.run_on_main_thread = num_workers == 0
        self.q = None
        self.stats_q = None
        self._started = False
        self.workers = []
        self.worker_instances = []
        if num_workers == 0:
            num_workers = 1  # The one worker is the main thread
        self.batch_size, self.input_maps, self.output_maps, self.num_workers, self.cache_size, self.weights, self.name, self.keep_paths = \
            batch_size, input_maps, output_maps, num_workers, cache_size, weights, name, keep_paths
        self.true_epochs = 0
        self.stats_string = ""
        if weights is None:
            worker_paths = np.array_split(paths, num_workers)
            self.true_epoch_lens = list(map(len, worker_paths))
            self.path_iters = [_ShufflePaths(p) for p in worker_paths]
        else:
            # split each path list into paths for each worker.
            # E.g. for two workers: [[p1, p2], [p3, p4, p5]] -> [[[p1], [p2]], [[p3, p4], [p5]]
            split_paths = [np.array_split(a, num_workers) for a in paths]
            # Next, each list of paths gets given to each worker. E.g. [[[p1], [p3, p4]], [[p2], [p5]]]
            worker_paths = np.swapaxes(split_paths, 0, 1)
            self.true_epoch_lens = [max(map(len, p)) for p in worker_paths]
            self.path_iters = [_WeightedPaths(p, weights) for p in worker_paths]

        self.batch_function_kwargs = {}
        if mixup_alpha > 0:
            self.batch_function = _mixup_batch
            self.batch_size *= 2
            self.batch_function_kwargs = {'alpha': mixup_alpha}
        elif siamese:
            self.batch_function = _identity_batch #_make_batch_siamese
        else:
            self.batch_function = _identity_batch

    def _init_workers(self):
        self.q = Queue(min(self.batch_size, TENSOR_GENERATOR_MAX_Q_SIZE))
        self.stats_q = Queue(len(self.worker_instances))
        self._started = True
        for i, (path_iter, iter_len) in enumerate(zip(self.path_iters, self.true_epoch_lens)):
            name = f'{self.name}_{i}'
            worker_instance = _MultiModalMultiTaskWorker(
                self.q,
                self.stats_q,
                self.num_workers,
                self.input_maps, self.output_maps,
                path_iter, iter_len,
                self.batch_function, self.batch_size, self.keep_paths, self.batch_function_kwargs,
                self.cache_size,
                name,
                self.augment,
            )
            self.worker_instances.append(worker_instance)
            if not self.run_on_main_thread:
                process = Process(
                    target=worker_instance.multiprocessing_worker, name=name,
                    args=(),
                )
                process.start()
                self.workers.append(process)
        logging.info(f"Started {i + 1} {self.name.replace('_', ' ')}s with cache size {self.cache_size/1e9}GB.")

    def set_worker_paths(self, paths: List[Path]):
        """In the single worker case, set the worker's paths."""
        if not self._started:
            self._init_workers()
        if not self.run_on_main_thread:
            raise ValueError('Cannot sort paths of multiprocessing workers. num_workers must be 0.')
        self.worker_instances[0].path_iter.paths = paths

    def __next__(self) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Optional[List[str]]]:
        if not self._started:
            self._init_workers()
        if self.stats_q.qsize() == self.num_workers:
            self.aggregate_and_print_stats()
        if self.run_on_main_thread:
            return next(self.worker_instances[0])
        else:
            return self.q.get(TENSOR_GENERATOR_TIMEOUT)

    def aggregate_and_print_stats(self):
        stats = Counter()
        self.true_epochs += 1
        cur_worker = 0
        while self.stats_q.qsize() != 0:
            cur_worker += 1
            worker_stats = self.stats_q.get().copy()
            for k in worker_stats:
                if stats[k] == 0 and cur_worker == 1 and ('_max' in k or '_min' in k):
                    stats[k] = worker_stats[k]
                elif '_max' in k:
                    stats[k] = max(stats[k], worker_stats[k])
                elif '_min' in k:
                    stats[k] = min(stats[k], worker_stats[k])
                else:
                    stats[k] += worker_stats[k]

        all_errors = [
            f'[{error}] - {count:.0f}'
            for error, count in sorted(stats.items(), key=lambda x: x[1], reverse=True) if 'Error' in error
        ]
        if len(all_errors) > 0:
            error_info = f'The following errors were raised:\n\t\t' + '\n\t\t'.join(all_errors)
        else:
            error_info = 'No errors raised.'

        eps = 1e-7
        for tm in self.input_maps + self.output_maps:
            if self.true_epochs != 1:
                break
            if tm.is_categorical() and tm.axes() == 1:
                n = stats[f'{tm.name}_n'] + eps
                self.stats_string = f'{self.stats_string}\nCategorical TensorMap: {tm.name} has {n:.0f} total examples.'
                for channel, index in tm.channel_map.items():
                    examples = stats[f'{tm.name}_index_{index:.0f}']
                    self.stats_string = f'{self.stats_string}\n\tLabel {channel} {examples} examples, {100 * (examples / n):0.2f}% of total.'
            elif tm.is_continuous() and tm.axes() == 1:
                sum_squared = stats[f'{tm.name}_sum_squared']
                n = stats[f'{tm.name}_n'] + eps
                n_sum = stats[f'{tm.name}_sum']
                mean = n_sum / n
                std = np.sqrt((sum_squared/n)-(mean*mean))
                self.stats_string = f'{self.stats_string}\nContinuous TensorMap: {tm.name} has {n:.0f} total examples.\n\tMean: {mean:0.2f}, '
                self.stats_string = f"{self.stats_string}Standard Deviation: {std:0.2f}, Max: {stats[f'{tm.name}_max']:0.2f}, Min: {stats[f'{tm.name}_min']:0.2f}"
            elif tm.is_time_to_event():
                sum_squared = stats[f'{tm.name}_sum_squared']
                n = stats[f'{tm.name}_n'] + eps
                n_sum = stats[f'{tm.name}_sum']
                mean = n_sum / n
                std = np.sqrt((sum_squared/n)-(mean*mean))
                self.stats_string = f"{self.stats_string}\nTime to event TensorMap: {tm.name} Total events: {stats[f'{tm.name}_events']}, "
                self.stats_string = f"{self.stats_string}\n\tMean Follow Up: {mean:0.2f}, Standard Deviation: {std:0.2f}, "
                self.stats_string = f"{self.stats_string}\n\tMax Follow Up: {stats[f'{tm.name}_max']:0.2f}, Min Follow Up: {stats[f'{tm.name}_min']:0.2f}"

        info_string = '\n\t'.join([
            f"Generator looped & shuffled over {sum(self.true_epoch_lens)} paths. Epoch: {self.true_epochs:.0f}",
            f"{stats['Tensors presented']:0.0f} tensors were presented.",
            f"{stats['skipped_paths']} paths were skipped because they previously failed.",
            f"{error_info}",
            f"{self.stats_string}",
        ])
        logging.info(f"\n!!!!>~~~~~~~~~~~~ {self.name} completed true epoch {self.true_epochs} ~~~~~~~~~~~~<!!!!\nAggregated information string:\n\t{info_string}")

    def kill_workers(self):
        if self._started and not self.run_on_main_thread:
            for worker in self.workers:
                worker.terminate()
            logging.info(f'Stopped {len(self.workers)} workers. {self.stats_string}')
        self.workers = []

    def __iter__(self):  # This is so python type annotations recognize TensorGenerator as an iterator
        return self

    def __del__(self):
        self.kill_workers()


class TensorMapArrayCache:
    """
    Caches numpy arrays created by tensor maps up to a maximum number of bytes
    """

    def __init__(self, max_size, input_tms: List[TensorMap], output_tms: List[TensorMap], max_rows: Optional[int] = np.inf):
        input_tms = [tm for tm in input_tms if tm.cacheable]
        output_tms = [tm for tm in output_tms if tm.cacheable]
        self.max_size = max_size
        self.data = {}
        self.row_size = sum(np.zeros(tm.static_shape(), dtype=np.float32).nbytes for tm in set(input_tms + output_tms))
        self.nrows = min(int(max_size / self.row_size), max_rows) if self.row_size else 0
        self.autoencode_names: Dict[str, str] = {}
        for tm in input_tms:
            self.data[tm.input_name()] = np.zeros((self.nrows,) + tm.static_shape(), dtype=np.float32)
        for tm in output_tms:
            if tm in input_tms:  # Useful for autoencoders
                self.autoencode_names[tm.output_name()] = tm.input_name()
            else:
                self.data[tm.output_name()] = np.zeros((self.nrows,) + tm.static_shape(), dtype=np.float32)
        self.files_seen = Counter()  # name -> max position filled in cache
        self.key_to_index = {}  # file_path, name -> position in self.data
        self.hits = 0
        self.failed_paths: Set[str] = set()

    def _fix_key(self, key: Tuple[str, str]) -> Tuple[str, str]:
        file_path, name = key
        return file_path, self.autoencode_names.get(name, name)

    def __setitem__(self, key: Tuple[str, str], value) -> bool:
        """
        :param key: should be a tuple file_path, name
        """
        file_path, name = self._fix_key(key)
        if key in self.key_to_index:  # replace existing value
            self.data[name][self.key_to_index[key]] = value
            return True
        if self.files_seen[name] >= self.nrows:  # cache already full
            return False
        self.key_to_index[key] = self.files_seen[name]
        self.data[name][self.key_to_index[key]] = value
        self.files_seen[name] += 1
        return True

    def __getitem__(self, key: Tuple[str, str]):
        """
        :param key: should be a tuple file_path, name
        """
        file_path, name = self._fix_key(key)
        val = self.data[name][self.key_to_index[file_path, name]]
        self.hits += 1
        return val

    def __contains__(self, key: Tuple[str, str]):
        return self._fix_key(key) in self.key_to_index

    def __len__(self):
        return sum(self.files_seen.values())

    def average_fill(self):
        return np.mean(list(self.files_seen.values()) or [0]) / self.nrows if self.nrows else 0

    def __str__(self):
        hits = f"The cache has had {self.hits} hits."
        fullness = ' - '.join(f"{name} has {count} / {self.nrows} tensors" for name, count in self.files_seen.items())
        return f'{hits} {fullness}.'


class _MultiModalMultiTaskWorker:

    def __init__(
        self,
        q: Queue,
        stats_q: Queue,
        num_workers: int,
        input_maps: List[TensorMap], output_maps: List[TensorMap],
        path_iter: PathIterator, true_epoch_len: int,
        batch_function: BatchFunction, batch_size: int, return_paths: bool, batch_func_kwargs: Dict,
        cache_size: float,
        name: str,
        augment: bool,
    ):
        self.q = q
        self.stats_q = stats_q
        self.num_workers = num_workers
        self.input_maps = input_maps
        self.output_maps = output_maps
        self.path_iter = path_iter
        self.true_epoch_len = true_epoch_len
        self.batch_function = batch_function
        self.batch_size = batch_size
        self.return_paths = return_paths
        self.batch_func_kwargs = batch_func_kwargs
        self.cache_size = cache_size
        self.name = name
        self.augment = augment

        self.stats = Counter()
        self.epoch_stats = Counter()
        self.start = time.time()
        self.paths_in_batch = []

        self.in_batch = {tm.input_name(): np.zeros((batch_size,) + tm.static_shape()) for tm in input_maps}
        self.out_batch = {tm.output_name(): np.zeros((batch_size,) + tm.static_shape()) for tm in output_maps}

        self.cache = TensorMapArrayCache(cache_size, input_maps, output_maps, true_epoch_len)
        self.dependents = {}
        self.idx = 0

    def _handle_tm(self, tm: TensorMap, is_input: bool, path: Path) -> h5py.File:
        name = tm.input_name() if is_input else tm.output_name()
        batch = self.in_batch if is_input else self.out_batch
        idx = self.stats['batch_index']

        if tm in self.dependents:
            batch[name][idx] = self.dependents[tm]
            if tm.cacheable:
                self.cache[path, name] = self.dependents[tm]
            self._collect_stats(tm, self.dependents[tm])
            return self.hd5
        if (path, name) in self.cache:
            batch[name][idx] = self.cache[path, name]
            return self.hd5
        if self.hd5 is None:  # Don't open hd5 if everything is in the self.cache
            self.hd5 = h5py.File(path, 'r')
        tensor = tm.postprocess_tensor(tm.tensor_from_file(tm, self.hd5, self.dependents), augment=self.augment, hd5=self.hd5)
        slices = tuple(slice(min(tm.static_shape()[i], tensor.shape[i])) for i in range(len(tensor.shape)))
        batch[name][(idx,)+slices] = tensor[slices]
        if tm.cacheable:
            self.cache[path, name] = batch[name][idx]
        self._collect_stats(tm, tensor)
        return self.hd5

    def _collect_stats(self, tm, tensor):
        if tm.is_time_to_event():
            self.epoch_stats[f'{tm.name}_events'] += tensor[0]
            self._collect_continuous_stats(tm, tensor[1])
        if tm.is_categorical() and tm.axes() == 1:
            self.epoch_stats[f'{tm.name}_index_{np.argmax(tensor):.0f}'] += 1
        if tm.is_continuous() and tm.axes() == 1:
            self._collect_continuous_stats(tm, tm.rescale(tensor)[0])
        self.epoch_stats[f'{tm.name}_n'] += 1

    def _collect_continuous_stats(self, tm, rescaled):
        if 0.0 == self.epoch_stats[f'{tm.name}_max'] == self.epoch_stats[f'{tm.name}_min']:
            self.epoch_stats[f'{tm.name}_max'] = rescaled
            self.epoch_stats[f'{tm.name}_min'] = rescaled
        self.epoch_stats[f'{tm.name}_max'] = max(rescaled, self.epoch_stats[f'{tm.name}_max'])
        self.epoch_stats[f'{tm.name}_min'] = min(rescaled, self.epoch_stats[f'{tm.name}_min'])
        self.epoch_stats[f'{tm.name}_sum'] += rescaled
        self.epoch_stats[f'{tm.name}_sum_squared'] += rescaled * rescaled

    def _handle_tensor_path(self, path: Path) -> None:
        hd5 = None
        if path in self.cache.failed_paths:
            self.epoch_stats['skipped_paths'] += 1
            return
        try:
            self.dependents = {}
            self.hd5 = None
            for tm in self.input_maps:
                hd5 = self._handle_tm(tm, True, path)
            for tm in self.output_maps:
                hd5 = self._handle_tm(tm, False, path)
            self.paths_in_batch.append(path)
            self.stats['Tensors presented'] += 1
            self.epoch_stats['Tensors presented'] += 1
            self.stats['batch_index'] += 1
        except (IndexError, KeyError, ValueError, OSError, RuntimeError) as e:
            error_name = type(e).__name__
            self.stats[f"{error_name} while attempting to generate tensor:\n{traceback.format_exc()}\n"] += 1
            self.epoch_stats[f"{error_name}: {e}"] += 1
            self.cache.failed_paths.add(path)
            _log_first_error(self.stats, path)
        finally:
            if hd5 is not None:
                hd5.close()

    def _on_epoch_end(self):
        self.stats['epochs'] += 1
        self.epoch_stats['epochs'] = self.stats['epochs']
        while self.stats_q.qsize() == self.num_workers:
            continue
        self.stats_q.put(self.epoch_stats)
        if self.stats['Tensors presented'] == 0:
            logging.error(f"Completed an epoch but did not find any tensors to yield")
        if 'test' in self.name:
            logging.warning(f'Test worker {self.name} completed a full epoch. Test results may be double counting samples.')
        self.start = time.time()
        self.epoch_stats = Counter()

    def multiprocessing_worker(self):
        for i, path in enumerate(self.path_iter):
            self._handle_tensor_path(path)
            if self.stats['batch_index'] == self.batch_size:

                out = self.batch_function(self.in_batch, self.out_batch, self.return_paths, self.paths_in_batch, **self.batch_func_kwargs)
                self.q.put(out)
                self.paths_in_batch = []
                self.stats['batch_index'] = 0
                self.in_batch = {tm.input_name(): np.zeros((self.batch_size,) + tm.static_shape()) for tm in self.input_maps}
                self.out_batch = {tm.output_name(): np.zeros((self.batch_size,) + tm.static_shape()) for tm in self.output_maps}
            if i > 0 and i % self.true_epoch_len == 0:
                self._on_epoch_end()

    def __next__(self):
        while self.stats['batch_index'] < self.batch_size:
            path = next(self.path_iter)
            self._handle_tensor_path(path)
            if self.idx > 0 and self.idx % self.true_epoch_len == 0:
                self._on_epoch_end()
            self.idx += 1
        self.stats['batch_index'] = 0
        out = self.batch_function(self.in_batch, self.out_batch, self.return_paths, self.paths_in_batch, **self.batch_func_kwargs)
        self.paths_in_batch = []
        return out


def big_batch_from_minibatch_generator(generator: TensorGenerator, minibatches: int,
                                       batch_size: int = 16, keep_paths: bool = False):
    """Collect minibatches into bigger batches

    Returns a dicts of numpy arrays like the same kind as generator but with more examples.

    Arguments:
        generator: TensorGenerator of minibatches
        minibatches: number of times to call generator and collect a minibatch

    Returns:
        A tuple of dicts mapping tensor names to big batches of numpy arrays mapping.
    """
    first_batch = next(iter(generator))
    #first_batch = generator.take(batch_size)
    saved_tensors = {}
    for key, batch_array in chain(first_batch[BATCH_INPUT_INDEX].items(), first_batch[BATCH_OUTPUT_INDEX].items()):
        shape = (batch_array.shape[0] * minibatches,) + batch_array.shape[1:]
        saved_tensors[key] = np.zeros(shape)
        batch_size = batch_array.shape[0]
        saved_tensors[key][:batch_size] = batch_array

    if keep_paths:
        paths = first_batch[BATCH_PATHS_INDEX]

    input_tensors, output_tensors = list(first_batch[BATCH_INPUT_INDEX]), list(first_batch[BATCH_OUTPUT_INDEX])
    for i in range(1, minibatches):
        logging.debug(f'big_batch_from_minibatch {100 * i / minibatches:.2f}% done.')
        next_batch = next(iter(generator))
        #next_batch = generator.take(batch_size)
        s, t = i * batch_size, (i + 1) * batch_size
        for key in input_tensors:
            saved_tensors[key][s:t] = next_batch[BATCH_INPUT_INDEX][key]
        for key in output_tensors:
            saved_tensors[key][s:t] = next_batch[BATCH_OUTPUT_INDEX][key]
        if keep_paths:
            paths.extend(next_batch[BATCH_PATHS_INDEX])

    for key, array in saved_tensors.items():
        logging.info(f"Made a big batch of tensors with key:{key} and shape:{array.shape}.")
    inputs = {key: saved_tensors[key] for key in input_tensors}
    outputs = {key: saved_tensors[key] for key in output_tensors}
    if keep_paths:
        return inputs, outputs, paths
    else:
        return inputs, outputs, None


def _get_train_valid_test_discard_ratios(
        valid_ratio: float,
        test_ratio: float,
        train_csv: str,
        valid_csv: str,
        test_csv: str,
) -> Tuple[int, int, int, int]:

    if valid_csv is not None:
        valid_ratio = 0
    if test_csv is not None:
        test_ratio = 0
    if train_csv is not None:
        train_ratio = 0
        discard_ratio = 1.0 - valid_ratio - test_ratio
    else:
        train_ratio = 1.0 - valid_ratio - test_ratio
        discard_ratio = 0

    if not math.isclose(train_ratio + valid_ratio + test_ratio + discard_ratio, 1.0):
        raise ValueError(f'ratios do not sum to 1, train/valid/test/discard = {train_ratio}/{valid_ratio}/{test_ratio}/{discard_ratio}')
    logging.debug(f'train/valid/test/discard ratios: {train_ratio}/{valid_ratio}/{test_ratio}/{discard_ratio}')

    return train_ratio, valid_ratio, test_ratio, discard_ratio


def _sample_csv_to_set(sample_csv: Optional[str] = None) -> Union[None, Set[str]]:
    if sample_csv is None:
        return None

    # Read CSV to dataframe and assume no header
    df = pd.read_csv(sample_csv, header=None)

    # If first row and column is castable to int, there is no header
    try:
        int(df.iloc[0].values[0])
    # If fails, must be header; overwrite column name with first row and remove first row
    except ValueError:
        df.columns = df.iloc[0]
        df = df[1:]

    # Declare set of possible MRN column names
    possible_mrn_col_names = {"sampleid", "medrecn", "mrn", "patient_id"}

    # Find intersection between CSV columns and possible MRN column names
    matches = set(df.columns).intersection(possible_mrn_col_names)

    # If no matches, assume the first column is MRN
    if not matches:
        mrn_col_name = df.columns[0]
    else:
         # Get first string from set of matches to use as column name
        mrn_col_name = next(iter(matches))

    if len(matches) > 1:
        logging.warning(
            f"{sample_csv} has more than one potential column for MRNs. Inferring most likely column name, but recommend explicitly setting MRN column name.",
        )

    # Isolate this column from the dataframe, and cast to strings
    sample_ids = df[mrn_col_name].apply(str)

    return set(sample_ids)


def get_train_valid_test_paths(
        tensors: str,
        sample_csv: str,
        valid_ratio: float,
        test_ratio: float,
        train_csv: str,
        valid_csv: str,
        test_csv: str,
) -> Tuple[List[str], List[str], List[str]]:
    """
    Return 3 disjoint lists of tensor paths.

    The paths are split in training, validation, and testing lists.
    If no arguments are given, paths are split into train/valid/test in the ratio 0.7/0.2/0.1.
    Otherwise, at least 2 arguments are required to specify train/valid/test sets.

    :param tensors: path to directory containing tensors
    :param sample_csv: path to csv containing sample ids, only consider sample ids for splitting
                       into train/valid/test sets if they appear in sample_csv
    :param valid_ratio: rate of tensors in validation list, mutually exclusive with valid_csv
    :param test_ratio: rate of tensors in testing list, mutually exclusive with test_csv
    :param train_csv: path to csv containing sample ids to reserve for training list
    :param valid_csv: path to csv containing sample ids to reserve for validation list, mutually exclusive with valid_ratio
    :param test_csv: path to csv containing sample ids to reserve for testing list, mutually exclusive with test_ratio

    :return: tuple of 3 lists of hd5 tensor file paths
    """
    train_paths = []
    valid_paths = []
    test_paths = []
    discard_paths = []

    train_ratio, valid_ratio, test_ratio, discard_ratio = _get_train_valid_test_discard_ratios(
        valid_ratio=valid_ratio,
        test_ratio=test_ratio,
        train_csv=train_csv,
        valid_csv=valid_csv,
        test_csv=test_csv,
    )

    choices = {
        'train': (train_paths, train_ratio),
        'valid': (valid_paths, valid_ratio),
        'test': (test_paths, test_ratio),
        'discard': (discard_paths, discard_ratio),
    }

    # parse csv's to disjoint sets, None if csv was None
    sample_set = _sample_csv_to_set(sample_csv)

    train_set = _sample_csv_to_set(train_csv)
    valid_set = _sample_csv_to_set(valid_csv)
    test_set = _sample_csv_to_set(test_csv)

    if train_set is not None and valid_set is not None and not train_set.isdisjoint(valid_set):
        raise ValueError('train and validation samples overlap')
    if train_set is not None and test_set is not None and not train_set.isdisjoint(test_set):
        raise ValueError('train and test samples overlap')
    if valid_set is not None and test_set is not None and not valid_set.isdisjoint(test_set):
        raise ValueError('validation and test samples overlap')

    # find tensors and split them among train/valid/test
    for root, dirs, files in os.walk(tensors):
        for name in files:
            path = os.path.join(root, name)
            split = os.path.splitext(name)
            sample_id = split[0]

            if split[-1].lower() != TENSOR_EXT:
                continue
            elif sample_set is not None and sample_id not in sample_set:
                continue
            elif train_set is not None and sample_id in train_set:
                train_paths.append(path)
            elif valid_set is not None and sample_id in valid_set:
                valid_paths.append(path)
            elif test_set is not None and sample_id in test_set:
                test_paths.append(path)
            else:
                choice = np.random.choice([k for k in choices], p=[choices[k][1] for k in choices])
                choices[choice][0].append(path)

    logging.info(f'Found {len(train_paths)} train, {len(valid_paths)} validation, and {len(test_paths)} testing tensors at: {tensors}')
    logging.debug(f'Discarded {len(discard_paths)} tensors due to given ratios')
    if len(train_paths) == 0 and len(valid_paths) == 0 and len(test_paths) == 0:
        raise ValueError(
            f'Not enough tensors at {tensors}\n'
            f'Found {len(train_paths)} training, {len(valid_paths)} validation, and {len(test_paths)} testing tensors\n'
            f'Discarded {len(discard_paths)} tensors',
        )

    return train_paths, valid_paths, test_paths


def get_train_valid_test_paths_split_by_csvs(
        tensors: str,
        balance_csvs: List[str],
        sample_csv: str,
        valid_ratio: float,
        test_ratio: float,
        train_csv: str,
        valid_csv: str,
        test_csv: str,
) -> Tuple[List[List[str]], List[List[str]], List[List[str]]]:
    stats = Counter()
    sample2group = {}
    for i, b_csv in enumerate(balance_csvs):
        lol = list(csv.reader(open(b_csv, 'r'), delimiter=','))
        logging.info(f"Class Balance CSV Header: {list(enumerate(lol[0]))}")

        for row in lol[1:]:
            sample_id = row[0]
            sample2group[sample_id] = i+1  # group 0 means background class
            stats['group_'+str(i+1)] += 1
    logging.info(f"Balancing with CSVs of Sample IDs stats: {stats}")

    train_paths = [[] for _ in range(len(balance_csvs)+1)]
    valid_paths = [[] for _ in range(len(balance_csvs)+1)]
    test_paths = [[] for _ in range(len(balance_csvs)+1)]

    _train, _valid, _test = get_train_valid_test_paths(
        tensors=tensors,
        sample_csv=sample_csv,
        valid_ratio=valid_ratio,
        test_ratio=test_ratio,
        train_csv=train_csv,
        valid_csv=valid_csv,
        test_csv=test_csv,
    )

    for paths, split_list in [(_train, train_paths), (_valid, valid_paths), (_test, test_paths)]:
        for path in paths:
            split = os.path.splitext(os.path.basename(path))
            sample_id = split[0]

            group = 0
            if sample_id in sample2group:
                group = sample2group[sample_id]
            split_list[group].append(path)

    for i in range(len(train_paths)):
        if len(train_paths[i]) == 0 or len(valid_paths[i]) == 0 or len(test_paths[i]) == 0:
            my_error = f"Not enough tensors at {tensors}\nGot {len(train_paths[i])} train {len(valid_paths[i])} valid and {len(test_paths[i])} test."
            raise ValueError(my_error)
        if i == 0:
            logging.info(f"Found {len(train_paths[i])} train {len(valid_paths[i])} valid and {len(test_paths[i])} test tensors outside the CSVs.")
        else:
            logging.info(f"CSV:{balance_csvs[i-1]}\nhas: {len(train_paths[i])} train, {len(valid_paths[i])} valid, {len(test_paths[i])} test tensors.")
    return train_paths, valid_paths, test_paths

def aug_model(rotation_factor, zoom_factor, translation_factor):
        rota = tf.keras.layers.RandomRotation(factor=rotation_factor, fill_mode='constant')

        zoom = tf.keras.layers.RandomZoom(
            height_factor=(-zoom_factor, zoom_factor),
            width_factor=None,
            fill_mode='constant',
        )

        trans = tf.keras.layers.RandomTranslation(
            height_factor=(-translation_factor, translation_factor),
            width_factor=(-translation_factor, translation_factor),
            fill_mode='constant',
        )

        layers = [rota, zoom, trans]
        aug_model = tf.keras.Sequential(layers)

        return aug_model

class ImageMaskAugmentor:
    """
    Applies random data augmentation (rotation, zoom and/or translation) to pairs of 2D images and segmentations.
    :param images: a dictionary mapping an input tensor map's name to an image tensor
    :param mask: a dictionary mapping an output tensor map's name to a segmentation tensor
    :param in_shapes: a dictionary mapping an input tensor map's name to its shape (including the batch_size)
    :param out_shapes: a dictionary mapping an output tensor map's name to its shape (including the batch_size)
    :param rotation_factor: a float represented as fraction of 2 Pi, e.g., rotation_factor = 0.014 results in an output rotated by a random amount in the range [-5 degrees, 5 degrees]
    :param zoom_factor: a float represented as fraction of value, e.g., zoom_factor = 0.05 results in an output zoomed in a random amount in the range [-5%, 5%]
    :param translation_factor: a float represented as a fraction of value, e.g., translation_factor = 0.05 results in an output shifted by a random amount in the range [-5%, 5%] in the x- and y- directions
    :return: an augmented image tensor and its corresponding augmented segmentation tensor
    """

    # Adapted from:
    # https://stackoverflow.com/questions/65475057/keras-data-augmentation-pipeline-for-image-segmentation-dataset-image-and-mask

    def __init__(self, rotation_factor: float, zoom_factor: float, translation_factor: float,
                 in_shapes: Dict[str, Tuple[int, int, int, int]],
                 out_shapes: Dict[str, Tuple[int, int, int, int]]):
        self.model = aug_model(rotation_factor, zoom_factor, translation_factor)
        self.in_shapes = in_shapes
        self.out_shapes = out_shapes

    def __call__(self, images: Dict[str, tf.Tensor], mask: Dict[str, tf.Tensor]) -> Tuple[tf.Tensor, tf.Tensor]:
        assert len(self.in_shapes) == 1
        assert len(self.out_shapes) == 1

        in_key = next(iter(self.in_shapes))
        out_key = next(iter(self.out_shapes))

        image_tensor = images[in_key]
        mask_tensor = mask[out_key]

        combined = tf.concat([image_tensor, mask_tensor], axis=-1)
        augmented = self.model(combined, training=True)

        input_channels = self.in_shapes[in_key][-1]
        image = augmented[..., :input_channels]
        mask = augmented[..., input_channels:]
        return image, mask

def test_train_valid_tensor_generators(
    tensor_maps_in: List[TensorMap],
    tensor_maps_out: List[TensorMap],
    tensor_maps_protected: List[TensorMap],
    tensors: str,
    batch_size: int,
    num_workers: int,
    training_steps: int,
    validation_steps: int,
    cache_size: float,
    balance_csvs: List[str],
    keep_paths: bool = False,
    keep_paths_test: bool = False,
    mixup_alpha: float = -1.0,
    sample_csv: str = None,
    valid_ratio: float = None,
    test_ratio: float = None,
    train_csv: str = None,
    valid_csv: str = None,
    test_csv: str = None,
    siamese: bool = False,
    rotation_factor: float = 0,
    zoom_factor: float = 0,
    translation_factor: float = 0,
    wrap_with_tf_dataset: bool = True,
    **kwargs
) -> Tuple[TensorGeneratorABC, TensorGeneratorABC, TensorGeneratorABC]:
    """ Get 3 tensor generator functions for training, validation and testing data.

    :param tensor_maps_in: list of TensorMaps that are input names to a model
    :param tensor_maps_out: list of TensorMaps that are output from a model
    :param tensor_maps_protected: list of TensorMaps that are sensitive to bias from a model
                                    only added to the test set
    :param tensors: directory containing tensors
    :param batch_size: number of examples in each mini-batch
    :param num_workers: number of processes spun off for training and testing. Validation uses half as many workers
    :param training_steps: Number of training batches that define a fake "epoch"
    :param validation_steps: Number of validation batches to create after each fake "epoch"
    :param cache_size: size in bytes of maximum cache for EACH worker
    :param balance_csvs: if not empty, generator will provide batches balanced amongst the Sample ID in these CSVs.
    :param keep_paths: also return the list of tensor files loaded for training and validation tensors
    :param keep_paths_test:  also return the list of tensor files loaded for testing tensors
    :param mixup_alpha: If positive, mixup batches and use this value as shape parameter alpha
    :param sample_csv: CSV file of sample ids, sample ids are considered for train/valid/test only if it is in sample_csv
    :param valid_ratio: rate of tensors to use for validation, mutually exclusive with valid_csv
    :param test_ratio: rate of tensors to use for testing, mutually exclusive with test_csv
    :param train_csv: CSV file of sample ids to use for training
    :param valid_csv: CSV file of sample ids to use for validation, mutually exclusive with valid_ratio
    :param test_csv: CSV file of sample ids to use for testing, mutually exclusive with test_ratio
    :param siamese: if True generate input for a siamese model i.e. a left and right input tensors for every input TensorMap
    :param rotation_factor: for data augmentation, a float represented as fraction of 2 Pi, e.g., rotation_factor = 0.014 results in an output rotated by a random amount in the range [-5 degrees, 5 degrees]
    :param zoom_factor: for data augmentation, a float represented as fraction of value, e.g., zoom_factor = 0.05 results in an output zoomed in a random amount in the range [-5%, 5%]
    :param translation_factor: for data augmentation, a float represented as a fraction of value, e.g., translation_factor = 0.05 results in an output shifted by a random amount in the range [-5%, 5%] in the x- and y- directions
    :param wrap_with_tf_dataset: if True will return tf.dataset objects for the 3 generators
    :return: A tuple of three generators. Each yields a Tuple of dictionaries of input and output numpy arrays for training, validation and testing.
    """
    generate_train, generate_valid, generate_test = None, None, None
    if len(balance_csvs) > 0:
        train_paths, valid_paths, test_paths = get_train_valid_test_paths_split_by_csvs(
            tensors=tensors,
            balance_csvs=balance_csvs,
            sample_csv=sample_csv,
            valid_ratio=valid_ratio,
            test_ratio=test_ratio,
            train_csv=train_csv,
            valid_csv=valid_csv,
            test_csv=test_csv,
        )
        weights = [1.0/(len(balance_csvs)+1) for _ in range(len(balance_csvs)+1)]
    else:
        train_paths, valid_paths, test_paths = get_train_valid_test_paths(
            tensors=tensors,
            sample_csv=sample_csv,
            valid_ratio=valid_ratio,
            test_ratio=test_ratio,
            train_csv=train_csv,
            valid_csv=valid_csv,
            test_csv=test_csv,
        )
        weights = None

    num_train_workers = int(training_steps / (training_steps + validation_steps) * num_workers) or (1 if num_workers else 0)
    num_valid_workers = int(validation_steps / (training_steps + validation_steps) * num_workers) or (1 if num_workers else 0)

    # use the longest list of [train_paths, valid_paths, test_paths], avoiding hard-coding one
    # in case it is empty
    paths = max([train_paths, valid_paths, test_paths], key=len)
    generator_class = pick_generator(paths, weights, mixup_alpha, siamese)

    generate_train = generator_class(
        batch_size=batch_size, input_maps=tensor_maps_in, output_maps=tensor_maps_out,
        paths=train_paths, num_workers=num_train_workers, cache_size=cache_size, weights=weights,
        keep_paths=keep_paths, mixup_alpha=mixup_alpha, name='train_worker', siamese=siamese, augment=True,
    )
    generate_valid = generator_class(
        batch_size=batch_size, input_maps=tensor_maps_in, output_maps=tensor_maps_out,
        paths=valid_paths, num_workers=num_valid_workers, cache_size=cache_size, weights=weights,
        keep_paths=keep_paths, mixup_alpha=0, name='validation_worker', siamese=siamese, augment=False,
    )
    generate_test = generator_class(
        batch_size=batch_size, input_maps=tensor_maps_in, output_maps=tensor_maps_out + tensor_maps_protected,
        paths=test_paths, num_workers=num_train_workers, cache_size=0, weights=weights,
        keep_paths=keep_paths or keep_paths_test, mixup_alpha=0, name='test_worker', siamese=siamese, augment=False,
    )

    do_augmentation = bool(rotation_factor or zoom_factor or translation_factor)
    if do_augmentation:
        logging.info(f'Augment with rotation {rotation_factor}, zoom {zoom_factor}, translation {translation_factor}')

    if do_augmentation:
        assert(len(tensor_maps_in) == 1, 'no support for multiple input tensors')
        assert(len(tensor_maps_out) == 1, 'no support for multiple output tensors')

    if wrap_with_tf_dataset and do_augmentation:
        in_shapes = {tm.input_name(): (batch_size,) + tm.static_shape() for tm in tensor_maps_in}
        out_shapes = {tm.output_name(): (batch_size,) + tm.static_shape() for tm in tensor_maps_out}

        train_dataset = tf.data.Dataset.from_generator(
            generate_train,
            output_types=({k: tf.float32 for k in in_shapes}, {k: tf.float32 for k in out_shapes}),
            output_shapes=(in_shapes, out_shapes),
        )
        augmentor = ImageMaskAugmentor(
            rotation_factor,
            zoom_factor,
            translation_factor,
            in_shapes,
            out_shapes
        )
        train_dataset = train_dataset.map(augmentor)

    if wrap_with_tf_dataset:
        in_shapes = {tm.input_name(): (batch_size,) + tm.static_shape() for tm in tensor_maps_in}
        out_shapes = {tm.output_name(): (batch_size,) + tm.static_shape() for tm in tensor_maps_out}

        train_dataset = tf.data.Dataset.from_generator(
            generate_train,
            output_types=({k: tf.float32 for k in in_shapes}, {k: tf.float32 for k in out_shapes}),
            output_shapes=(in_shapes, out_shapes),
        )
        valid_dataset = tf.data.Dataset.from_generator(
            generate_valid,
            output_types=({k: tf.float32 for k in in_shapes}, {k: tf.float32 for k in out_shapes}),
            output_shapes=(in_shapes, out_shapes),
        )
        test_dataset = tf.data.Dataset.from_generator(
            generate_test,
            output_types=({k: tf.float32 for k in in_shapes}, {k: tf.float32 for k in out_shapes}),
            output_shapes=(in_shapes, out_shapes),
        )

    if wrap_with_tf_dataset:
        return train_dataset, valid_dataset, test_dataset
    elif do_augmentation:
        return train_dataset, generate_valid, generate_test
    else:
        return generate_train, generate_valid, generate_test

def _log_first_error(stats: Counter, tensor_path: str):
    for k in stats:
        if 'Error' in k and stats[k] == 1:
            stats[k] += 1  # Increment so we only see these messages once
            logging.debug(f"At tensor path: {tensor_path}")
            logging.debug(f"Got first error: {k}")


def _identity_batch(in_batch: Batch, out_batch: Batch, return_paths: bool, paths: List[Path]):
    return (in_batch, out_batch, paths) if return_paths else (in_batch, out_batch)


def _mixup_batch(in_batch: Batch, out_batch: Batch, return_paths: bool, paths: List[Path], alpha: float = 1.0, permute_first: bool = False):
    full_batch = in_batch.values().__iter__().__next__().shape[0]
    half_batch = full_batch // 2

    if permute_first:
        permuted = np.random.permutation(full_batch)
        for k in in_batch:
            in_batch[k] = in_batch[k][permuted, ...]
        for k in out_batch:
            out_batch[k] = out_batch[k][permuted, ...]

    mixed_ins = {k: np.zeros((half_batch,) + in_batch[k].shape[1:]) for k in in_batch}
    mixed_outs = {k: np.zeros((half_batch,) + out_batch[k].shape[1:]) for k in out_batch}
    for i in range(half_batch):
        weight0 = np.random.beta(alpha, alpha)
        weight1 = 1 - weight0
        for k in in_batch:
            mixed_ins[k][i] = (in_batch[k][i, ...] * weight0) + (in_batch[k][half_batch + i, ...] * weight1)
        for k in out_batch:
            mixed_outs[k][i] = (out_batch[k][i, ...] * weight0) + (out_batch[k][half_batch + i, ...] * weight1)

    return _identity_batch(mixed_ins, mixed_outs, return_paths, paths[:half_batch])


def _make_batch_siamese(in_batch: Batch, out_batch: Batch, return_paths: bool, paths: List[Path]):
    full_batch = in_batch.values().__iter__().__next__().shape[0]
    half_batch = full_batch // 2

    siamese_in = {k+'_left': np.zeros((half_batch,) + in_batch[k].shape[1:]) for k in in_batch}
    siamese_in.update({k+'_right': np.zeros((half_batch,) + in_batch[k].shape[1:]) for k in in_batch})
    siamese_out = {'output_siamese': np.zeros((half_batch, 1))}

    for i in range(half_batch):
        for k in in_batch:
            siamese_in[k+'_left'][i] = in_batch[k][i, ...]
            siamese_in[k+'_right'][i] = in_batch[k][half_batch + i, ...]
        random_task_key = np.random.choice(list(out_batch.keys()))
        siamese_out['output_siamese'][i] = 0 if np.array_equal(out_batch[random_task_key][i], out_batch[random_task_key][i+half_batch]) else 1

    return _identity_batch(siamese_in, siamese_out, return_paths, paths)


def _weighted_batch(in_batch: Batch, out_batch: Batch, return_paths: bool, paths: List[Path]):
    return (in_batch, out_batch, paths) if return_paths else (in_batch, out_batch)


def pad_1d(list_of_arrays, max_len, pad_value=0, dtype='int32'):
    out = np.full((len(list_of_arrays), max_len), pad_value, dtype=dtype)
    for i, a in enumerate(list_of_arrays):
        L = min(len(a), max_len)
        out[i, :L] = a[:L]
    return out


def pad_2d(list_of_arrays, max_len, feat, pad_value=0.0, dtype='float32'):
    out = np.full((len(list_of_arrays), max_len, feat), pad_value, dtype=dtype)
    for i, a in enumerate(list_of_arrays):
        L = min(len(a), max_len)
        out[i, :L, :] = a[:L]
    return out


def make_ds(Xv, Xn, m, y, w, BATCH, shuffle=False):
    if Xv is not None:
        ds = tf.data.Dataset.from_tensor_slices((
            {'view': Xv, 'num': Xn, 'mask': m},
            y,
            w
        ))
    else:
        ds = tf.data.Dataset.from_tensor_slices((
            {'num': Xn, 'mask': m},
            y,
            w
        ))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(m), seed=42)
    return ds.batch(BATCH).prefetch(tf.data.AUTOTUNE)


def build_datasets(
        df,
        INPUT_NUMERIC_COLS,
        input_categorical_column,
        REGRESSION_TARGETS,
        BINARY_TARGETS,
        AGGREGATE_COLUMN,
        sort_column,
        MAX_LEN,
        BATCH,
):
    TARGETS_ALL = REGRESSION_TARGETS + BINARY_TARGETS
    # ---------- Checks ----------
    required_cols = set([AGGREGATE_COLUMN] + INPUT_NUMERIC_COLS + TARGETS_ALL)
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"df is missing required columns: {missing}")

    # ============ Build MRN sequences ============
    # Encode view_prediction to ids (0=PAD)
    if input_categorical_column:
        view_vocab = pd.Series(df[input_categorical_column].astype(str).unique())
        view2id = {v: i + 1 for i, v in enumerate(view_vocab)}  # 0 reserved for PAD
        df['_view_id'] = df[input_categorical_column].astype(str).map(view2id).fillna(0).astype(int)

    df = df.sort_values([AGGREGATE_COLUMN, sort_column], ascending=[True, True]).reset_index(drop=True)

    grouped = df.groupby(AGGREGATE_COLUMN, sort=False)
    mrn_list = list(grouped.groups.keys())

    # Log number of groups found
    logging.info(f"Found {len(mrn_list)} groups (unique {AGGREGATE_COLUMN}s)")

    # Collect sequences & MRN-level targets
    seq_view_ids = []
    seq_numeric = []
    seq_len = []
    y_dict = {t: [] for t in TARGETS_ALL}

    for mrn, g in grouped:
        if input_categorical_column:
            seq_view_ids.append(g['_view_id'].values.astype('int32'))
        seq_numeric.append(g[INPUT_NUMERIC_COLS].values.astype('float32'))  # (L, F)
        seq_len.append(len(g))

        # One MRN-level label per task (take first; or assert all equal)
        for t in TARGETS_ALL:
            y_dict[t].append(g[t].iloc[0] if t in g.columns else np.nan)

    seq_len = np.array(seq_len, dtype='int32')
    N = len(seq_len)
    Feat = len(INPUT_NUMERIC_COLS)

    # Log sequence length statistics
    logging.info(f"Sequence length statistics across {N} groups:")
    logging.info(f"  Mean: {seq_len.mean():.2f}")
    logging.info(f"  Std:  {seq_len.std():.2f}")
    logging.info(f"  Min:  {seq_len.min()}")
    logging.info(f"  Max:  {seq_len.max()}")
    logging.info(f"  Median: {np.median(seq_len):.2f}")

    if input_categorical_column:
        X_view = pad_1d(seq_view_ids, MAX_LEN, pad_value=0, dtype='int32')  # (N, T)
    X_num = pad_2d(seq_numeric, MAX_LEN, Feat, pad_value=0.0, dtype='float32')  # (N, T, F)
    mask_bt = (np.arange(MAX_LEN)[None, :] < seq_len[:, None])  # (N, T) True=real

    # Targets as arrays + sample weights (1 if present, 0 if NaN)
    y_arrays = {}
    sw_arrays = {}
    for t in TARGETS_ALL:
        y = np.array(y_dict[t], dtype='float32')
        sw = (~pd.Series(y).isna()).astype('float32').values
        # replace NaN with 0 to keep shape; weights will drop them
        y[np.isnan(y)] = 0.0
        y_arrays[t] = y
        sw_arrays[t] = sw

    # Log summary statistics for each target
    logging.info("Target summary statistics:")
    for t in TARGETS_ALL:
        y = y_arrays[t]
        sw = sw_arrays[t]
        valid_mask = sw > 0
        if valid_mask.sum() > 0:
            valid_y = y[valid_mask]
            logging.info(f"  {t}:")
            logging.info(f"    Valid samples: {valid_mask.sum()}/{len(y)} ({valid_mask.mean()*100:.1f}%)")
            logging.info(f"    Mean: {valid_y.mean():.4f}")
            logging.info(f"    Std:  {valid_y.std():.4f}")
            logging.info(f"    Min:  {valid_y.min():.4f}")
            logging.info(f"    Max:  {valid_y.max():.4f}")
        else:
            logging.info(f"  {t}: No valid samples")

    # ============ Train/Val split by MRN ============
    idx_train, idx_val = train_test_split(np.arange(N), test_size=0.2, random_state=42)

    def sel(idx):
        if input_categorical_column:
            Xv = X_view[idx]
        else:
            Xv = None
        Xn = X_num[idx]
        m = mask_bt[idx]
        ys = {t: y_arrays[t][idx] for t in TARGETS_ALL}
        sw = {t: sw_arrays[t][idx] for t in TARGETS_ALL}
        return Xv, Xn, m, ys, sw

    Xv_tr, Xn_tr, m_tr, y_tr, w_tr = sel(idx_train)
    Xv_va, Xn_va, m_va, y_va, w_va = sel(idx_val)

    train_ds = make_ds(Xv_tr, Xn_tr, m_tr, y_tr, w_tr, BATCH, shuffle=True)
    val_ds = make_ds(Xv_va, Xn_va, m_va, y_va, w_va, BATCH, shuffle=False)
    return train_ds, val_ds



def df_to_datasets_from_generator(df, INPUT_NUMERIC_COLS, input_categorical_column, AGGREGATE_COLUMN, sort_column,
                                  sort_column_ascend, TARGETS_ALL, MAX_LEN, BATCH, train_csv, valid_csv, test_csv):
    if input_categorical_column:
        view_vocab = pd.Series(df[input_categorical_column].astype(str).unique())
        view2id = {v: i + 1 for i, v in enumerate(view_vocab)}  # 0 reserved for PAD
        df['_view_id'] = df[input_categorical_column].astype(str).map(view2id).fillna(0).astype(int)
    # Reproducible ordering
    df_sorted = df.sort_values([AGGREGATE_COLUMN, sort_column],
                               ascending=[True, sort_column_ascend]).reset_index(drop=True)

    # ----- Train/Val/Test split by MRN based on CSV files -----
    group_ids = df_sorted[AGGREGATE_COLUMN].drop_duplicates().to_numpy()

    # Log number of groups found
    logging.info(f"Found {len(group_ids)} groups (unique {AGGREGATE_COLUMN}s)")

    # Get unique MRNs from the dataframe
    unique_mrns = df_sorted[AGGREGATE_COLUMN].drop_duplicates().to_numpy()
    logging.info(f"Found {len(unique_mrns)} unique MRNs in dataframe")

    # Check if CSV files are provided
    if train_csv or valid_csv or test_csv:
        # Read MRNs from CSV files
        train_mrns = _sample_csv_to_set(train_csv) if train_csv else set()
        valid_mrns = _sample_csv_to_set(valid_csv) if valid_csv else set()
        test_mrns = _sample_csv_to_set(test_csv) if test_csv else set()

        logging.info(f"CSV files contain: {len(train_mrns)} train MRNs, {len(valid_mrns)} valid MRNs, {len(test_mrns)} test MRNs")

        # Log sample MRNs for debugging
        if len(unique_mrns) > 0:
            logging.info(f"Sample dataframe MRNs: {list(unique_mrns[:3])}")
        if len(train_mrns) > 0:
            logging.info(f"Sample train CSV MRNs: {list(list(train_mrns)[:3])}")

        # Split MRNs into train/val/test based on CSV membership
        train_mrn_set = set()
        val_mrn_set = set()
        test_mrn_set = set()

        for mrn in unique_mrns:
            mrn_str = str(mrn)
            if train_mrns and mrn_str in train_mrns:
                train_mrn_set.add(mrn)
            elif valid_mrns and mrn_str in valid_mrns:
                val_mrn_set.add(mrn)
            elif test_mrns and mrn_str in test_mrns:
                test_mrn_set.add(mrn)

        logging.info(f"Matched MRNs: {len(train_mrn_set)} train, {len(val_mrn_set)} valid, {len(test_mrn_set)} test")
    else:
        # No CSV files provided - randomly split MRNs: 80% train, 10% valid, 10% test
        logging.info("No CSV files provided. Randomly splitting MRNs: 80% train, 10% valid, 10% test")

        from sklearn.model_selection import train_test_split

        # First split: 80% train, 20% temp (for valid+test)
        train_mrns_arr, temp_mrns = train_test_split(
            unique_mrns, test_size=0.2, random_state=42
        )

        # Second split: split temp into 50% valid, 50% test (each 10% of total)
        val_mrns_arr, test_mrns_arr = train_test_split(
            temp_mrns, test_size=0.5, random_state=42
        )

        train_mrn_set = set(train_mrns_arr)
        val_mrn_set = set(val_mrns_arr)
        test_mrn_set = set(test_mrns_arr)

        logging.info(f"Random split MRNs: {len(train_mrn_set)} train, {len(val_mrn_set)} valid, {len(test_mrn_set)} test")

    # Now map group_ids to train/val/test based on their MRN
    # Build a mapping from group_id to mrn
    group_to_mrn = df_sorted.groupby(AGGREGATE_COLUMN)[AGGREGATE_COLUMN].first().to_dict()

    train_ids = set()
    val_ids = set()
    test_ids = set()

    for gid in group_ids:
        mrn = group_to_mrn.get(gid)
        if mrn in train_mrn_set:
            train_ids.add(gid)
        elif mrn in val_mrn_set:
            val_ids.add(gid)
        elif mrn in test_mrn_set:
            test_ids.add(gid)

    # Log training, validation, and test set sizes
    train_groups = len(train_ids)
    val_groups = len(val_ids)
    test_groups = len(test_ids)

    # Calculate total rows for train, validation, and test sets
    train_rows = 0
    val_rows = 0
    test_rows = 0
    for gid, g in df_sorted.groupby(AGGREGATE_COLUMN, sort=False):
        if gid in train_ids:
            train_rows += len(g)
        elif gid in val_ids:
            val_rows += len(g)
        elif gid in test_ids:
            test_rows += len(g)

    logging.info(f"Training set: {train_groups} groups, {train_rows} total rows")
    logging.info(f"Validation set: {val_groups} groups, {val_rows} total rows")
    logging.info(f"Test set: {test_groups} groups, {test_rows} total rows")

    # Validate that we have at least some data in train set
    if train_groups == 0:
        if train_csv or valid_csv or test_csv:
            raise ValueError(
                f"Training set is empty! No MRNs from CSV files matched the dataframe. "
                f"Check that MRN formats match between CSV and dataframe. "
                f"Dataframe has {len(unique_mrns)} unique MRNs."
            )
        else:
            raise ValueError(
                f"Training set is empty! Dataframe has {len(unique_mrns)} unique MRNs but none were assigned to training."
            )

    Feat = len(INPUT_NUMERIC_COLS)

    # ---------- Build once (unchanged index, no view used) ----------
    group_index = {}
    seq_lengths = []
    for gid, g in df_sorted.groupby(AGGREGATE_COLUMN, sort=False):
        first = g.index[0]
        last = g.index[-1]
        group_index[gid] = (first, last)  # inclusive row-span within df_sorted
        seq_lengths.append(len(g))

    # Log sequence length statistics
    seq_lengths = np.array(seq_lengths)
    logging.info(f"Sequence length statistics across {len(seq_lengths)} groups:")
    logging.info(f"  Mean: {seq_lengths.mean():.2f}")
    logging.info(f"  Std:  {seq_lengths.std():.2f}")
    logging.info(f"  Min:  {seq_lengths.min()}")
    logging.info(f"  Max:  {seq_lengths.max()}")
    logging.info(f"  Median: {np.median(seq_lengths):.2f}")

    # ----- tf.data from_generator + padded_batch (no giant tensors) -----
    feature_sig = {
        'num' : tf.TensorSpec(shape=(None, Feat), dtype=tf.float32),
        'mask': tf.TensorSpec(shape=(None,), dtype=tf.bool),
    }
    if input_categorical_column:
        feature_sig['view'] = tf.TensorSpec(shape=(None,), dtype=tf.int32)
    label_sig = {t: tf.TensorSpec(shape=(), dtype=tf.float32) for t in TARGETS_ALL}
    weight_sig = {t: tf.TensorSpec(shape=(), dtype=tf.float32) for t in TARGETS_ALL}

    # ---------- Generator WITHOUT VIEW_COL ----------
    def group_generator(selected_ids):
        # Preload numeric block for fast slicing
        arr_num = df_sorted[INPUT_NUMERIC_COLS].to_numpy(np.float32)
        if input_categorical_column:
            arr_view = df_sorted['_view_id'].to_numpy(np.int32)

        # MRN-level targets (max of non-NA values per group); None if target missing in df
        arr_tgts = {
            t: (df_sorted.groupby(AGGREGATE_COLUMN)[t].max() if t in df_sorted.columns else None)
            for t in TARGETS_ALL
        }

        # Log summary statistics for each target (only once per generator call)
        if not hasattr(group_generator, '_logged_targets'):
            logging.info("Target summary statistics:")
            for t in TARGETS_ALL:
                if arr_tgts[t] is not None:
                    target_values = arr_tgts[t].values
                    valid_mask = ~pd.isna(target_values)
                    if valid_mask.sum() > 0:
                        valid_values = target_values[valid_mask]
                        logging.info(f"  {t}:")
                        logging.info(f"    Valid samples: {valid_mask.sum()}/{len(target_values)} ({valid_mask.mean()*100:.1f}%)")
                        logging.info(f"    Mean: {valid_values.mean():.4f}")
                        logging.info(f"    Std:  {valid_values.std():.4f}")
                        logging.info(f"    Min:  {valid_values.min():.4f}")
                        logging.info(f"    Max:  {valid_values.max():.4f}")
                    else:
                        logging.info(f"  {t}: No valid samples")
                else:
                    logging.info(f"  {t}: Target not found in dataframe")
            group_generator._logged_targets = True

        for gid in selected_ids:
            span = group_index.get(gid)
            if span is None:
                continue
            start, last = span
            end = last + 1

            # Features: ONLY numeric + mask (truncated to MAX_LEN if needed)
            num = arr_num[start:end, :]  # (T, F)
            if input_categorical_column:
                view = arr_view[start:end]  # (T,)
            T = num.shape[0]

            # Truncate to MAX_LEN if sequence is longer
            if T > MAX_LEN:
                num = num[:MAX_LEN, :]
                if input_categorical_column:
                    view = view[:MAX_LEN]
                T = MAX_LEN

            mask = np.ones((T,), dtype=bool)  # (T,)

            # Labels + sample weights (one scalar per task)
            y, sw = {}, {}
            for t in TARGETS_ALL:
                if arr_tgts[t] is not None:
                    v = arr_tgts[t].get(gid, np.nan)
                    has = not pd.isna(v)
                    y[t] = np.float32(v if has else 0.0)
                    sw[t] = np.float32(1.0 if has else 0.0)
                else:
                    y[t] = np.float32(0.0)
                    sw[t] = np.float32(0.0)

            if input_categorical_column:
                yield {'view': view, 'num': num, 'mask': mask}, y, sw
            else:
                yield {'num': num, 'mask': mask}, y, sw

    def make_tf_dataset_from_generator(id_set, shuffle=False):
        ds = tf.data.Dataset.from_generator(
            lambda: group_generator(id_set),
            output_signature=(feature_sig, label_sig, weight_sig)
        )
        if shuffle and len(id_set) > 0:
            ds = ds.shuffle(buffer_size=len(id_set), reshuffle_each_iteration=True)
        # Pad sequences to max length *in the batch* (capped at MAX_LEN):
        if 'view' in feature_sig:
            ds = ds.padded_batch(
                BATCH,
                padded_shapes=(
                    {'view': [MAX_LEN], 'num': [MAX_LEN, Feat], 'mask': [MAX_LEN]},
                    {t: [] for t in TARGETS_ALL},
                    {t: [] for t in TARGETS_ALL},
                ),
                padding_values=(
                    {'view': np.int32(0), 'num': np.float32(0.0), 'mask': False},
                    {t: np.float32(0.0) for t in TARGETS_ALL},   # labels
                    {t: np.float32(0.0) for t in TARGETS_ALL},   # weights
                ),
                drop_remainder=False,
            )
        else:
            ds = ds.padded_batch(
                BATCH,
                padded_shapes=(
                    {'num': [MAX_LEN, Feat], 'mask': [MAX_LEN]},
                    {t: [] for t in TARGETS_ALL},
                    {t: [] for t in TARGETS_ALL},
                ),
                padding_values=(
                    {'num': np.float32(0.0), 'mask': False},
                    {t: np.float32(0.0) for t in TARGETS_ALL},   # labels
                    {t: np.float32(0.0) for t in TARGETS_ALL},   # weights
                ),
                drop_remainder=False,
            )
        return ds.prefetch(tf.data.AUTOTUNE).repeat()

    train_ds = make_tf_dataset_from_generator(train_ids, shuffle=True)
    val_ds = make_tf_dataset_from_generator(val_ids, shuffle=False)
    test_ds = make_tf_dataset_from_generator(test_ids, shuffle=False)
    return train_ds, val_ds, test_ds


class LongitudinalDataloader:
    def __init__(
        self,
        input_file_path,
        label_file_path,
        group_column,
        sort_column,
        latent_dim,
        numeric_columns,
        categorical_columns,
        label_columns,
        max_seq_len,
        batch_size,
        shuffle=True,
        sort_column_ascend=True,
        train_csv: Optional[str] = None,
        valid_csv: Optional[str] = None,
        test_csv: Optional[str] = None,
    ):
        self.input_ds = ds.dataset(input_file_path, format="parquet")
        self.label_ds = (
            self.input_ds
            if label_file_path in [None, input_file_path]
            else ds.dataset(label_file_path, format="parquet")
        )

        self.latent_cols = [f"latent_{i}" for i in range(latent_dim)]
        self.latent_dim = latent_dim
        self.numeric_columns = numeric_columns
        self.categorical_columns = categorical_columns
        self.label_columns = label_columns
        self.max_seq_len = max_seq_len
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.group_column = group_column
        self.train_csv = train_csv
        self.valid_csv = valid_csv
        self.test_csv = test_csv

        # ---- build MRN → row indices ONCE ----
        index_tbl = self.input_ds.to_table(
            columns=[group_column, sort_column]
        ).to_pandas()

        index_tbl = index_tbl.sort_values([group_column, sort_column], ascending=[True, sort_column_ascend], kind="mergesort")

        self.group_to_indices = {}
        self.group_ids = []
        for group_id, g in index_tbl.groupby(group_column, sort=False):
            idxs = g.index.values
            if max_seq_len:
                idxs = idxs[-max_seq_len:]
            self.group_to_indices[group_id] = idxs
            self.group_ids.append(group_id)

        # Log number of groups found
        logging.info(f"Found {len(self.group_ids)} groups (unique {group_column}s)")

        # Log sequence length statistics
        seq_lengths = np.array([len(idxs) for idxs in self.group_to_indices.values()])
        logging.info(f"Sequence length statistics across {len(seq_lengths)} groups:")
        logging.info(f"  Mean: {seq_lengths.mean():.2f}")
        logging.info(f"  Std:  {seq_lengths.std():.2f}")
        logging.info(f"  Min:  {seq_lengths.min()}")
        logging.info(f"  Max:  {seq_lengths.max()}")
        logging.info(f"  Median: {np.median(seq_lengths):.2f}")

        # ------------------------------------------------------------
        # Fallback cache (contiguous range cache, bounded by RAM)
        # ------------------------------------------------------------
        avail = psutil.virtual_memory().available
        # conservative: allow cache to use ~40% of RAM
        self._fallback_cache_rows = int(
            0.4 * avail / (self.latent_dim * 4 + 16)
        )

        @lru_cache(maxsize=8)
        def _cached_take(start, length):
            idxs = np.arange(start, start + length)
            tbl = self.input_ds.take(idxs)

            lat = np.stack(
                [tbl.column(c).to_numpy(zero_copy_only=False) for c in self.latent_cols],
                axis=1,
            ).astype(np.float32)

            num = {
                c: tbl.column(c).to_numpy(zero_copy_only=False).astype(np.float32)
                for c in self.numeric_columns
            }

            cat = {
                c: tbl.column(c).to_numpy(zero_copy_only=False).astype(np.int32)
                for c in self.categorical_columns
            }

            return lat, num, cat

        self._cached_take = _cached_take
        # Log summary statistics for each target
        if label_columns:
            logging.info("Target summary statistics:")
            label_tbl = self.label_ds.to_table(columns=label_columns).to_pandas()
            for t in label_columns:
                if t in label_tbl.columns:
                    target_values = label_tbl[t].values
                    valid_mask = ~pd.isna(target_values)
                    if valid_mask.sum() > 0:
                        valid_values = target_values[valid_mask]
                        logging.info(f"  {t}:")
                        logging.info(f"    Valid samples: {valid_mask.sum()}/{len(target_values)} ({valid_mask.mean()*100:.1f}%)")
                        logging.info(f"    Mean: {np.mean(valid_values):.4f}")
                        logging.info(f"    Std:  {np.std(valid_values):.4f}")
                        logging.info(f"    Min:  {np.min(valid_values):.4f}")
                        logging.info(f"    Max:  {np.max(valid_values):.4f}")
                    else:
                        logging.info(f"  {t}: No valid samples")
                else:
                    logging.info(f"  {t}: Target not found in dataset")
    def _try_build_in_memory_dataset(self, groups):
        num_groups = len(groups)

        bytes_per_step = (
            self.latent_dim * 4
            + len(self.numeric_columns) * 4
            + len(self.categorical_columns) * 4
            + 1
        )
        est_bytes = (
            num_groups * self.max_seq_len * bytes_per_step * 2.0
        )

        avail = psutil.virtual_memory().available
        if est_bytes > 0.6 * avail:
            return None

        # ---- allocate tensors ----
        X_latent = np.zeros(
            (num_groups, self.max_seq_len, self.latent_dim), dtype=np.float32
        )
        X_mask = np.zeros((num_groups, self.max_seq_len), dtype=bool)

        X_num = {
            c: np.zeros((num_groups, self.max_seq_len), dtype=np.float32)
            for c in self.numeric_columns
        }
        X_cat = {
            c: np.zeros((num_groups, self.max_seq_len), dtype=np.int32)
            for c in self.categorical_columns
        }

        Y = {c: np.zeros((num_groups,), dtype=np.float32) for c in self.label_columns}
        SW = {c: np.zeros((num_groups,), dtype=np.float32) for c in self.label_columns}

        # ---- load all rows ONCE ----
        flat_idxs = np.concatenate(groups)
        inp = self.input_ds.take(flat_idxs)
        lab = self.label_ds.take(flat_idxs)

        lat_all = np.stack(
            [inp.column(c).to_numpy(zero_copy_only=False) for c in self.latent_cols],
            axis=1,
        ).astype(np.float32)

        num_all = {
            c: inp.column(c).to_numpy(zero_copy_only=False).astype(np.float32)
            for c in self.numeric_columns
        }

        cat_all = {
            c: inp.column(c).to_numpy(zero_copy_only=False).astype(np.int32)
            for c in self.categorical_columns
        }

        lab_all = np.stack(
            [lab.column(c).to_numpy(zero_copy_only=False) for c in self.label_columns],
            axis=1,
        ).astype(np.float32)

        offset = 0
        for i, idxs in enumerate(groups):
            T = len(idxs)
            sl = slice(offset, offset + T)

            X_latent[i, :T] = lat_all[sl]
            X_mask[i, :T] = True

            for c in self.numeric_columns:
                X_num[c][i, :T] = num_all[c][sl]

            for c in self.categorical_columns:
                X_cat[c][i, :T] = cat_all[c][sl]

            vals = lab_all[sl][-1]
            for j, c in enumerate(self.label_columns):
                if not np.isnan(vals[j]):
                    Y[c][i] = float(vals[j])
                    SW[c][i] = 1.0

            offset += T

        X = {
            "latent": X_latent,
            "mask": X_mask,
            **{f"num_{c}": X_num[c] for c in self.numeric_columns},
            **{f"cat_{c}": X_cat[c] for c in self.categorical_columns},
        }

        ds_tf = tf.data.Dataset.from_tensor_slices((X, Y, SW))
        if self.shuffle:
            ds_tf = ds_tf.shuffle(num_groups)

        return ds_tf.batch(self.batch_size).prefetch(tf.data.AUTOTUNE)
    # ------------------------------------------------------------
    # Original generator, now with cached fallback
    # ------------------------------------------------------------
    def _generator(self, groups=None):
        
        #groups = list(self.group_to_indices.values())
        if self.shuffle:
            np.random.shuffle(groups)

        from more_itertools import chunked

        for chunk in chunked(groups, 256):
            flat_idxs = np.concatenate(chunk)

            flat_idxs = np.asarray(flat_idxs)
            start = int(flat_idxs.min())
            length = int(flat_idxs.ptp()) + 1

            # Heuristic: only use contiguous caching if it's not too sparse.
            # density = how many rows we actually need / span we would read
            density = flat_idxs.size / float(length)

            USE_CONTIG_CACHE = (density >= 0.25) and (length <= 200_000)  # tune if you want

            if USE_CONTIG_CACHE:
                lat_blk, num_blk, cat_blk = self._cached_take(start, length)
                rel = flat_idxs - start

                lat_all = lat_blk[rel]
                num_all = {c: num_blk[c][rel] for c in self.numeric_columns}
                cat_all = {c: cat_blk[c][rel] for c in self.categorical_columns}
            else:
                # SAFE PATH (no huge span reads)
                inp = self.input_ds.take(flat_idxs)

                lat_all = np.stack(
                    [inp.column(c).to_numpy(zero_copy_only=False) for c in self.latent_cols],
                    axis=1,
                ).astype(np.float32)

                num_all = {
                    c: inp.column(c).to_numpy(zero_copy_only=False).astype(np.float32)
                    for c in self.numeric_columns
                }

                cat_all = {
                    c: inp.column(c).to_numpy(zero_copy_only=False).astype(np.int32)
                    for c in self.categorical_columns
                }
                
            lab = self.label_ds.take(flat_idxs)
            lab_all = np.stack(
                [
                    lab.column(c).to_numpy(zero_copy_only=False)
                    for c in self.label_columns
                ],
                axis=1,
            ).astype(np.float32)

            offset = 0
            for idxs in chunk:
                T = len(idxs)
                sl = slice(offset, offset + T)

                x = {
                    "latent": lat_all[sl],
                    "mask": np.ones(T, dtype=bool),
                }

                for c in self.numeric_columns:
                    x[f"num_{c}"] = num_all[c][sl]

                for c in self.categorical_columns:
                    x[f"cat_{c}"] = cat_all[c][sl]

                vals = lab_all[sl][-1]
                y, sw = {}, {}
                for i, c in enumerate(self.label_columns):
                    if np.isnan(vals[i]):
                        y[c] = 0.0
                        sw[c] = 0.0
                    else:
                        y[c] = float(vals[i])
                        sw[c] = 1.0

                yield x, y, sw
                offset += T

    # ------------------------------------------------------------
    # Dataset builder (Option 5 → fallback)
    # ------------------------------------------------------------
    def get_tf_dataset(self, selected_group_ids=None):
        groups = (
                    [self.group_to_indices[g] for g in selected_group_ids]
                    if selected_group_ids is not None
                    else list(self.group_to_indices.values())
                )
        ds = self._try_build_in_memory_dataset(groups)
        if ds is not None:
            print("Using in-memory dataset.")
            return ds
        output_signature = (
            {
                "latent": tf.TensorSpec((None, self.latent_dim), tf.float32),
                "mask": tf.TensorSpec((None,), tf.bool),
                **{
                    f"num_{c}": tf.TensorSpec((None,), tf.float32)
                    for c in self.numeric_columns
                },
                **{
                    f"cat_{c}": tf.TensorSpec((None,), tf.int32)
                    for c in self.categorical_columns
                },
            },
            {c: tf.TensorSpec((), tf.float32) for c in self.label_columns},
            {c: tf.TensorSpec((), tf.float32) for c in self.label_columns},
        )

        ds_tf = tf.data.Dataset.from_generator(
            lambda: self._generator(groups),
            output_signature=output_signature,
        )

        return ds_tf.padded_batch(
            self.batch_size,
            padded_shapes=(
                {
                    "latent": [self.max_seq_len, self.latent_dim],
                    "mask": [self.max_seq_len],
                    **{f"num_{c}": [self.max_seq_len] for c in self.numeric_columns},
                    **{
                        f"cat_{c}": [self.max_seq_len] for c in self.categorical_columns
                    },
                },
                {c: [] for c in self.label_columns},
                {c: [] for c in self.label_columns},
            ),
        ).repeat().prefetch(tf.data.AUTOTUNE)

    # ------------------------------------------------------------
    # Train / val split unchanged
    # ------------------------------------------------------------
    def get_train_valid_test_datasets(self, valid_frac=0.1, test_frac=0.1, seed=42):
        """Split data into train, validation, and test datasets.

        If train_csv, valid_csv, or test_csv were provided during initialization,
        those will be used to determine the splits. Otherwise, random splitting
        is performed based on the provided fractions.

        Args:
            valid_frac: Fraction of data for validation (used only if valid_csv not provided)
            test_frac: Fraction of data for testing (used only if test_csv not provided)
            seed: Random seed for reproducibility

        Returns:
            Tuple of (train_dataset, valid_dataset, test_dataset)
        """
        unique_group_ids = set(self.group_ids)

        # Check if CSV files are provided
        if self.train_csv or self.valid_csv or self.test_csv:
            # Read group IDs from CSV files
            train_set = _sample_csv_to_set(self.train_csv) if self.train_csv else set()
            valid_set = _sample_csv_to_set(self.valid_csv) if self.valid_csv else set()
            test_set = _sample_csv_to_set(self.test_csv) if self.test_csv else set()

            logging.info(f"CSV files contain: {len(train_set)} train IDs, {len(valid_set)} valid IDs, {len(test_set)} test IDs")

            # Log sample IDs for debugging
            if len(unique_group_ids) > 0:
                sample_ids = list(unique_group_ids)[:3]
                logging.info(f"Sample dataset {self.group_column}s: {sample_ids}")
            if len(train_set) > 0:
                logging.info(f"Sample train CSV IDs: {list(train_set)[:3]}")

            # Split group IDs into train/val/test based on CSV membership
            train_ids = set()
            val_ids = set()
            test_ids = set()

            for gid in unique_group_ids:
                gid_str = str(gid)
                if train_set and gid_str in train_set:
                    train_ids.add(gid)
                elif valid_set and gid_str in valid_set:
                    val_ids.add(gid)
                elif test_set and gid_str in test_set:
                    test_ids.add(gid)

            logging.info(f"Matched IDs: {len(train_ids)} train, {len(val_ids)} valid, {len(test_ids)} test")
        else:
            # No CSV files provided - randomly split
            logging.info(f"No CSV files provided. Randomly splitting: {(1-valid_frac-test_frac)*100:.0f}% train, {valid_frac*100:.0f}% valid, {test_frac*100:.0f}% test")

            rng = np.random.default_rng(seed)
            group_ids_array = np.array(list(unique_group_ids))
            rng.shuffle(group_ids_array)

            n = len(group_ids_array)
            test_split = int(test_frac * n)
            valid_split = int((test_frac + valid_frac) * n)

            test_ids = set(group_ids_array[:test_split])
            val_ids = set(group_ids_array[test_split:valid_split])
            train_ids = set(group_ids_array[valid_split:])

            logging.info(f"Random split IDs: {len(train_ids)} train, {len(val_ids)} valid, {len(test_ids)} test")

        # Calculate total rows for train, validation, and test sets
        train_rows = sum(len(self.group_to_indices[gid]) for gid in train_ids)
        val_rows = sum(len(self.group_to_indices[gid]) for gid in val_ids)
        test_rows = sum(len(self.group_to_indices[gid]) for gid in test_ids)

        logging.info(f"Training set: {len(train_ids)} groups, {train_rows} total rows")
        logging.info(f"Validation set: {len(val_ids)} groups, {val_rows} total rows")
        logging.info(f"Test set: {len(test_ids)} groups, {test_rows} total rows")

        # Create datasets with appropriate shuffle settings
        train_loader = copy.copy(self)
        val_loader = copy.copy(self)
        test_loader = copy.copy(self)

        train_loader.shuffle = self.shuffle
        val_loader.shuffle = False
        test_loader.shuffle = False

        train_ds = train_loader.get_tf_dataset(selected_group_ids=list(train_ids))
        val_ds = val_loader.get_tf_dataset(selected_group_ids=list(val_ids))
        test_ds = test_loader.get_tf_dataset(selected_group_ids=list(test_ids))

        return train_ds, val_ds, test_ds

class _QueueIterator:
    def __init__(self, loader, q):
        self.loader = loader
        self.q = q

        self._chunk = None
        self._lat_all = None
        self._num_all = None
        self._cat_all = None
        self._lab_all = None
        self._cum_lengths = None
        self._group_idx = 0
        self._lock = threading.Lock()


    def __iter__(self):
        return self

    def __next__(self):
        with self._lock:
            # If no chunk loaded or exhausted → load next
            if self._chunk is None or self._group_idx >= len(self._chunk):

                chunk, inp, lab = self.q.get()

                # Convert Arrow → NumPy ONCE per chunk
                n = inp.num_rows
                self._lat_all = np.empty((n, self.loader.latent_dim), dtype=np.float32)

                for i, c in enumerate(self.loader.latent_cols):
                    self._lat_all[:, i] = inp.column(c).to_numpy(zero_copy_only=False)

                self._num_all = {
                    c: inp.column(c).to_numpy(zero_copy_only=False).astype(np.float32)
                    for c in self.loader.numeric_columns
                }

                self._cat_all = {
                    c: inp.column(c).to_numpy(zero_copy_only=False).astype(np.int32)
                    for c in self.loader.categorical_columns
                }

                self._lab_all = np.stack(
                    [lab.column(c).to_numpy(zero_copy_only=False)
                    for c in self.loader.label_columns],
                    axis=1,
                ).astype(np.float32)

                self._chunk = chunk
                self._cum_lengths = np.cumsum([0] + [len(g) for g in chunk])
                self._group_idx = 0

            # Emit one group
            start = self._cum_lengths[self._group_idx]
            end = self._cum_lengths[self._group_idx + 1]

            T = end - start
            sl = slice(start, end)

            x = {
                "latent": self._lat_all[sl],
                "mask": np.ones(T, dtype=bool),
            }

            for c in self.loader.numeric_columns:
                x[f"num_{c}"] = self._num_all[c][sl]

            for c in self.loader.categorical_columns:
                x[f"cat_{c}"] = self._cat_all[c][sl]

            vals = self._lab_all[sl][-1]
            y, sw = {}, {}

            for i, c in enumerate(self.loader.label_columns):
                if np.isnan(vals[i]):
                    y[c] = 0.0
                    sw[c] = 0.0
                else:
                    y[c] = float(vals[i])
                    sw[c] = 1.0

            self._group_idx += 1
            return x, y, sw


class LongitudinalDataloaderFast:
    def __init__(
        self,
        input_file_path,
        label_file_path,
        group_column,
        sort_column,
        latent_dim,
        numeric_columns,
        categorical_columns,
        label_columns,
        max_seq_len,
        batch_size,
        shuffle=True,
        sort_column_ascend=True,
        train_csv: Optional[str] = None,
        valid_csv: Optional[str] = None,
        test_csv: Optional[str] = None,
    ):
        self.input_ds = ds.dataset(input_file_path, format="parquet")
        self.label_ds = (
            self.input_ds
            if label_file_path in [None, input_file_path]
            else ds.dataset(label_file_path, format="parquet")
        )
        self._inp_pf = pq.ParquetFile(input_file_path)
        self._lab_pf = self._inp_pf if label_file_path in [None, input_file_path] else pq.ParquetFile(label_file_path)

        self._rg_sizes = [self._inp_pf.metadata.row_group(i).num_rows for i in range(self._inp_pf.num_row_groups)]
        print("Mean row group size:", np.mean(self._rg_sizes))
        print("Max row group size:", np.max(self._rg_sizes))

        self._rg_cum = np.cumsum([0] + self._rg_sizes).astype(np.int64)  # len = num_row_groups+1


        self.latent_cols = [f"latent_{i}" for i in range(latent_dim)]
        self.latent_dim = latent_dim
        self.numeric_columns = numeric_columns
        self.categorical_columns = categorical_columns
        self.label_columns = label_columns
        self.max_seq_len = max_seq_len
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.group_column = group_column
        self.train_csv = train_csv
        self.valid_csv = valid_csv
        self.test_csv = test_csv

        # ---- build MRN → row indices ONCE ----
        index_tbl = self.input_ds.to_table(
            columns=[group_column, sort_column]
        ).to_pandas()

        index_tbl = index_tbl.sort_values([group_column, sort_column], ascending=[True, sort_column_ascend], kind="mergesort")

        self._q = queue.Queue(maxsize=2)   # you can tune this, but it isn't the main fix
        self._producer_started = False
        self._consumer_iter = _QueueIterator(self, self._q)

        self.group_to_indices = {}
        self.group_ids = []
        for group_id, g in index_tbl.groupby(group_column, sort=False):
            idxs = g.index.values
            if max_seq_len:
                idxs = idxs[-max_seq_len:]
            self.group_to_indices[group_id] = idxs
            self.group_ids.append(group_id)

        # Log number of groups found
        logging.info(f"Found {len(self.group_ids)} groups (unique {group_column}s)")

        self._groups_by_locality = sorted(
            list(self.group_to_indices.values()),
            key=lambda idxs: int(np.median(idxs))
        )


        # Log sequence length statistics
        seq_lengths = np.array([len(idxs) for idxs in self.group_to_indices.values()])
        logging.info(f"Sequence length statistics across {len(seq_lengths)} groups:")
        logging.info(f"  Mean: {seq_lengths.mean():.2f}")
        logging.info(f"  Std:  {seq_lengths.std():.2f}")
        logging.info(f"  Min:  {seq_lengths.min()}")
        logging.info(f"  Max:  {seq_lengths.max()}")
        logging.info(f"  Median: {np.median(seq_lengths):.2f}")

        # Log summary statistics for each target
        if label_columns:
            logging.info("Target summary statistics:")
            label_tbl = self.label_ds.to_table(columns=label_columns).to_pandas()
            for t in label_columns:
                if t in label_tbl.columns:
                    target_values = label_tbl[t].values
                    valid_mask = ~pd.isna(target_values)
                    if valid_mask.sum() > 0:
                        valid_values = target_values[valid_mask]
                        logging.info(f"  {t}:")
                        logging.info(f"    Valid samples: {valid_mask.sum()}/{len(target_values)} ({valid_mask.mean()*100:.1f}%)")
                        logging.info(f"    Mean: {np.mean(valid_values):.4f}")
                        logging.info(f"    Std:  {np.std(valid_values):.4f}")
                        logging.info(f"    Min:  {np.min(valid_values):.4f}")
                        logging.info(f"    Max:  {np.max(valid_values):.4f}")
                    else:
                        logging.info(f"  {t}: No valid samples")
                else:
                    logging.info(f"  {t}: Target not found in dataset")
    
    #@lru_cache(maxsize=1)
    def _read_row_group_cached(self, rg: int, cols_key: str):
        cols = cols_key.split("|") if cols_key else None
        return self._inp_pf.read_row_group(rg, columns=cols)

    #@lru_cache(maxsize=1)
    def _read_row_group_cached_labels(self, rg: int, cols_key: str):
        cols = cols_key.split("|") if cols_key else None
        return self._lab_pf.read_row_group(rg, columns=cols)
    
    def _gather_parquet_rows(self, flat_idxs: np.ndarray, columns, is_label=False):
        """
        Gather rows from Parquet using row-group aware reads.
        flat_idxs: global row indices (np.int64)
        columns: list[str] columns to read
        returns: pyarrow.Table with rows in the SAME order as flat_idxs
        """
        flat_idxs = np.asarray(flat_idxs, dtype=np.int64)
        n = flat_idxs.size
        out_pos = np.arange(n, dtype=np.int64)

        # Map global idx -> row group id (vectorized)
        # rg is in [0, num_row_groups-1]
        rg = np.searchsorted(self._rg_cum[1:], flat_idxs, side="right").astype(np.int32)
        local = (flat_idxs - self._rg_cum[rg]).astype(np.int64)

        # Group requests by row group (preserve original positions)
        order = np.argsort(rg, kind="mergesort")
        rg_s = rg[order]
        local_s = local[order]
        pos_s = out_pos[order]

        cols_key = "|".join(columns)

        chunks = []
        start = 0
        while start < n:
            r = int(rg_s[start])
            end = start + 1
            while end < n and rg_s[end] == r:
                end += 1

            local_idx = local_s[start:end]
            pos_idx = pos_s[start:end]

            if is_label:
                tbl = self._read_row_group_cached_labels(r, cols_key)
            else:
                tbl = self._read_row_group_cached(r, cols_key)

            # take within row group, then scatter back to original order
            taken = tbl.take(pa.array(local_idx))
            # attach position so we can re-order later
            taken = taken.append_column("__pos__", pa.array(pos_idx))

            chunks.append(taken)
            start = end

        merged = pa.concat_tables(chunks)

        # reorder by __pos__ to match original flat_idxs order
        pos = merged["__pos__"].to_numpy(zero_copy_only=False)
        back = np.argsort(pos, kind="mergesort")
        merged = merged.take(pa.array(back)).drop(["__pos__"])

        return merged

    def _start_producer(self, groups):
        if self._producer_started:
            return
        self._producer_started = True

        from more_itertools import chunked
        import gc

        def producer_loop():
            # IMPORTANT: never exit; keep feeding forever
            while True:
                # NOTE: do NOT reshuffle if you don't need to
                if self.shuffle:
                    np.random.shuffle(groups)

                groups_local = self._groups_by_locality
                n = len(groups_local)

                # optionally shuffle window order by shuffling indices
                window_starts = list(range(0, n, 5000))
                if self.shuffle:
                    np.random.shuffle(window_starts)

                for start in window_starts:
                    end = min(start + 5000, n)
                    window = groups_local[start:end]
                    if self.shuffle:
                        np.random.shuffle(window)

                    for chunk in chunked(window, 256):
                #for chunk in chunked(groups, 256):  # tune chunk size later
                        flat_idxs = np.concatenate(chunk).astype(np.int64)

                        # <-- this is your fast row-group aware gather version -->
                        inp_cols = self.latent_cols + self.numeric_columns + self.categorical_columns
                        lab_cols = self.label_columns

                        inp = self._gather_parquet_rows(flat_idxs, inp_cols, is_label=False)
                        lab = self._gather_parquet_rows(flat_idxs, lab_cols, is_label=(self._lab_pf is not self._inp_pf))

                        self._q.put((chunk, inp, lab))

        threading.Thread(target=producer_loop, daemon=True).start()

    # ------------------------------------------------------------
    # Dataset builder (Option 5 → fallback)
    # ------------------------------------------------------------
    def get_tf_dataset(self, selected_group_ids=None):
        groups = (
                    [self.group_to_indices[g] for g in selected_group_ids]
                    if selected_group_ids is not None
                    else list(self.group_to_indices.values())
                )
        
        output_signature = (
            {
                "latent": tf.TensorSpec((None, self.latent_dim), tf.float32),
                "mask": tf.TensorSpec((None,), tf.bool),
                **{
                    f"num_{c}": tf.TensorSpec((None,), tf.float32)
                    for c in self.numeric_columns
                },
                **{
                    f"cat_{c}": tf.TensorSpec((None,), tf.int32)
                    for c in self.categorical_columns
                },
            },
            {c: tf.TensorSpec((), tf.float32) for c in self.label_columns},
            {c: tf.TensorSpec((), tf.float32) for c in self.label_columns},
        )
        self._start_producer(groups)

        ds_tf = tf.data.Dataset.from_generator(
            lambda: self._consumer_iter,   # <-- persistent iterator, not a new generator
            output_signature=output_signature,
        )
        ds_tf = ds_tf.repeat()
        return ds_tf.padded_batch(
            self.batch_size,
            padded_shapes=(
                {
                    "latent": [self.max_seq_len, self.latent_dim],
                    "mask": [self.max_seq_len],
                    **{f"num_{c}": [self.max_seq_len] for c in self.numeric_columns},
                    **{
                        f"cat_{c}": [self.max_seq_len] for c in self.categorical_columns
                    },
                },
                {c: [] for c in self.label_columns},
                {c: [] for c in self.label_columns},
            ),
        ).prefetch(tf.data.AUTOTUNE)

    # ------------------------------------------------------------
    # Train / val split unchanged
    # ------------------------------------------------------------
    def get_train_valid_test_datasets(self, valid_frac=0.1, test_frac=0.1, seed=42):
        """Split data into train, validation, and test datasets.

        If train_csv, valid_csv, or test_csv were provided during initialization,
        those will be used to determine the splits. Otherwise, random splitting
        is performed based on the provided fractions.

        Args:
            valid_frac: Fraction of data for validation (used only if valid_csv not provided)
            test_frac: Fraction of data for testing (used only if test_csv not provided)
            seed: Random seed for reproducibility

        Returns:
            Tuple of (train_dataset, valid_dataset, test_dataset)
        """
        unique_group_ids = set(self.group_ids)

        # Check if CSV files are provided
        if self.train_csv or self.valid_csv or self.test_csv:
            # Read group IDs from CSV files
            train_set = _sample_csv_to_set(self.train_csv) if self.train_csv else set()
            valid_set = _sample_csv_to_set(self.valid_csv) if self.valid_csv else set()
            test_set = _sample_csv_to_set(self.test_csv) if self.test_csv else set()

            logging.info(f"CSV files contain: {len(train_set)} train IDs, {len(valid_set)} valid IDs, {len(test_set)} test IDs")

            # Log sample IDs for debugging
            if len(unique_group_ids) > 0:
                sample_ids = list(unique_group_ids)[:3]
                logging.info(f"Sample dataset {self.group_column}s: {sample_ids}")
            if len(train_set) > 0:
                logging.info(f"Sample train CSV IDs: {list(train_set)[:3]}")

            # Split group IDs into train/val/test based on CSV membership
            train_ids = set()
            val_ids = set()
            test_ids = set()

            for gid in unique_group_ids:
                gid_str = str(gid)
                if train_set and gid_str in train_set:
                    train_ids.add(gid)
                elif valid_set and gid_str in valid_set:
                    val_ids.add(gid)
                elif test_set and gid_str in test_set:
                    test_ids.add(gid)

            logging.info(f"Matched IDs: {len(train_ids)} train, {len(val_ids)} valid, {len(test_ids)} test")
        else:
            # No CSV files provided - randomly split
            logging.info(f"No CSV files provided. Randomly splitting: {(1-valid_frac-test_frac)*100:.0f}% train, {valid_frac*100:.0f}% valid, {test_frac*100:.0f}% test")

            rng = np.random.default_rng(seed)
            group_ids_array = np.array(list(unique_group_ids))
            rng.shuffle(group_ids_array)

            n = len(group_ids_array)
            test_split = int(test_frac * n)
            valid_split = int((test_frac + valid_frac) * n)

            test_ids = set(group_ids_array[:test_split])
            val_ids = set(group_ids_array[test_split:valid_split])
            train_ids = set(group_ids_array[valid_split:])

            logging.info(f"Random split IDs: {len(train_ids)} train, {len(val_ids)} valid, {len(test_ids)} test")

        # Calculate total rows for train, validation, and test sets
        train_rows = sum(len(self.group_to_indices[gid]) for gid in train_ids)
        val_rows = sum(len(self.group_to_indices[gid]) for gid in val_ids)
        test_rows = sum(len(self.group_to_indices[gid]) for gid in test_ids)

        logging.info(f"Training set: {len(train_ids)} groups, {train_rows} total rows")
        logging.info(f"Validation set: {len(val_ids)} groups, {val_rows} total rows")
        logging.info(f"Test set: {len(test_ids)} groups, {test_rows} total rows")

        # Create datasets with appropriate shuffle settings
        train_loader = copy.copy(self)
        val_loader = copy.copy(self)
        test_loader = copy.copy(self)

        train_loader.shuffle = self.shuffle
        val_loader.shuffle = False
        test_loader.shuffle = False

        train_ds = train_loader.get_tf_dataset(selected_group_ids=list(train_ids))
        val_ds = val_loader.get_tf_dataset(selected_group_ids=list(val_ids))
        test_ds = test_loader.get_tf_dataset(selected_group_ids=list(test_ids))

        return train_ds, val_ds, test_ds