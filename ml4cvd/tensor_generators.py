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
import h5py
import time
import logging
import traceback
import numpy as np
from collections import Counter
from random import choices, shuffle
from contextlib import contextmanager
from multiprocessing import Process, Queue
from typing import List, Dict, Tuple, Generator, Optional


from ml4cvd.defines import TENSOR_EXT
from ml4cvd.TensorMap import TensorMap

np.set_printoptions(threshold=np.inf)


def _partition_list(a: List, n: int) -> List[List]:
    """
    Partitions list into almost even chunks, with the last chunk getting len(a) % n extra parts
    :param a: list to partition
    :param n: number of partitions
    :return: list of partitioned lists
    """
    partitions = []
    step = len(a) // n
    for i in range(n):
        if i == n - 1:
            partitions.append(a[i * step:])
        else:
            partitions.append(a[i * step: (i + 1) * step])
    return partitions


class TensorGenerator:
    def __init__(self, batch_size, input_maps, output_maps, paths, num_workers, cache_size, weights=None, keep_paths=False, mixup=0.0, name='worker'):
        """
        :param paths: If weights is provided, paths should be a list of path lists the same length as weights
        """
        self.workers = []
        max_q_size = 32
        self.q = Queue(max_q_size)
        if weights is None:
            worker_path_lists = _partition_list(paths, num_workers)
        else:
            worker_path_lists = [_partition_list(a, num_workers) for a in paths]
        for i, worker_paths in enumerate(worker_path_lists):
            process = Process(target=multimodal_multitask_worker, name=f'{name}_{i}',
                              args=(self.q, batch_size, input_maps, output_maps, worker_paths, keep_paths, cache_size, mixup, weights))
            self.workers.append(process)
        self._started = False

    def __next__(self) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Optional[List[str]]]:
        if not self._started:
            self._started = True
            for process in self.workers:
                process.start()
        logging.debug(f'Currently there are {self.q.qsize()} queued batches.')
        return self.q.get()

    def kill(self):
        if self._started:
            for p in self.workers:
                p.terminate()


class TMArrayCache:
    """
    Caches numpy arrays created by tensor maps up to a maximum number of bytes
    """

    def __init__(self, max_size, input_tms: List[TensorMap], output_tms: List[TensorMap]):
        self.max_size = max_size
        self.data = {}
        self.row_size = sum(np.zeros(tm.shape, dtype=np.float32).nbytes for tm in input_tms + output_tms)
        self.nrows = int(max_size / self.row_size)
        for tm in input_tms:
            self.data[tm.input_name()] = np.zeros((self.nrows,) + tm.shape, dtype=np.float32)
        for tm in output_tms:
            self.data[tm.output_name()] = np.zeros((self.nrows,) + tm.shape, dtype=np.float32)
        self.files_seen = Counter()  # name -> max position filled in cache
        self.key_to_index = {}  # file_path, name -> position in self.data
        self.hits = 0

    def __setitem__(self, key: Tuple[str, str], value):
        """
        :param key: should be a tuple file_path, name
        """
        file_path, name = key
        if key in self.key_to_index:
            self.data[name][self.key_to_index[key]] = value
            return True
        if self.files_seen[name] >= self.nrows:
            return False
        self.key_to_index[key] = self.files_seen[name]
        self.data[name][self.key_to_index[key]] = value
        self.files_seen[name] += 1
        return True

    def __getitem__(self, item: Tuple[str, str]):
        """
        :param item: should be a tuple file_path, name
        """
        file_path, name = item
        val = self.data[name][self.key_to_index[file_path, name]]
        self.hits += 1
        return val

    def __contains__(self, item: Tuple[str, str]):
        return item in self.key_to_index

    def __len__(self):
        return sum(self.files_seen.values())

    def average_fill(self):
        return np.mean(list(self.files_seen.values())) / self.nrows


def _handle_tm(tm: TensorMap, hd5: h5py.File, cache: TMArrayCache, is_input: bool, tp: str, idx, batch, dependents):
    name = tm.input_name() if is_input else tm.output_name()
    if tm in dependents:  # for now, TMAPs with dependents are not cacheable
        batch[name][idx] = dependents[tm]
        return hd5
    if (tp, name) in cache:
        batch[name][idx] = cache[tp, name]
        return hd5
    if hd5 is None:  # Don't open hd5 if everything is in the cache
        hd5 = h5py.File(tp, 'r')
    dependents = {}
    tensor = tm.tensor_from_file(tm, hd5, dependents)
    batch[name][idx] = tensor
    if tm.cacheable:
        cache[tp, name] = tensor
    return hd5


def multimodal_multitask_worker(q: Queue, batch_size: int, input_maps, output_maps, train_paths, keep_paths, cache_size,
                                mixup_alpha, weights=None):
    """Generalized data generator of input and output tensors for feed-forward networks.

    The `modes` are the different inputs, and the `tasks` are given by the outputs.
    Infinitely loops over all examples yielding batch_size input and output tensor mappings.

    Arguments:
        q: Multiprocessing queue to push finished batches to
        batch_size: number of examples in each minibatch
        input_maps: list of TensorMaps that are input names to a model
        output_maps: list of TensorMaps that are output from a model
        train_paths: list of hd5 tensors shuffled after every loop or epoch
        keep_paths: If true will also yield the paths to the tensors used in this batch
        mixup_alpha: If positive, mixup batches and use this value as shape parameter alpha

    What gets enqueued:
        A tuple of dicts for the tensor mapping tensor names to numpy arrays
        {input_1:input_tensor1, input_2:input_tensor2, ...}, {output_name1:output_tensor1, ...}
        if include_paths_in_batch is True the 3rd element of tuple will be the list of paths
    """
    assert len(train_paths) > 0

    stats = Counter()
    paths_in_batch = []
    if mixup_alpha > 0:
        batch_size *= 2
    in_batch = {tm.input_name(): np.zeros((batch_size,)+tm.shape) for tm in input_maps}
    out_batch = {tm.output_name(): np.zeros((batch_size,)+tm.shape) for tm in output_maps}
    cache = TMArrayCache(cache_size, input_maps, output_maps)  # 1 GB, will be param later

    while True:
        simple_stats = Counter()
        start = time.time()
        if weights is not None:
            if len(weights) != len(train_paths):
                raise ValueError('weights must be the same length as train_paths.')
            epoch_len = max((len(p) for p in train_paths))
            paths = []
            for path_list, weight in zip(train_paths, weights):
                paths += choices(path_list, k=int(weight * epoch_len))
            shuffle(paths)
        paths = train_paths
        for tp in paths:
            try:
                hd5 = None
                dependents = {}
                for tm in input_maps:
                    hd5 = _handle_tm(tm, hd5, cache, True, tp, stats['batch_index'], in_batch, dependents)
                for tm in output_maps:
                    hd5 = _handle_tm(tm, hd5, cache, False, tp, stats['batch_index'], out_batch, dependents)
                paths_in_batch.append(tp)
                stats['batch_index'] += 1
                stats['Tensors presented'] += 1
                if stats['batch_index'] == batch_size:
                    if mixup_alpha > 0 and keep_paths:
                        q.put((_mixup_batch(in_batch, out_batch, mixup_alpha) + (paths_in_batch[:batch_size//2],)))
                    elif mixup_alpha > 0:
                        q.put(_mixup_batch(in_batch, out_batch, mixup_alpha))
                    elif keep_paths:
                        q.put((in_batch, out_batch, paths_in_batch))
                    else:
                        q.put((in_batch, out_batch))
                    stats['batch_index'] = 0
                    paths_in_batch = []
            except IndexError as e:
                stats[f"IndexError while attempting to generate tensor:\n{traceback.format_exc()}\n"] += 1
                simple_stats[str(e)] += 1
            except KeyError as e:
                stats[f"KeyError while attempting to generate tensor:\n{traceback.format_exc()}\n"] += 1
                simple_stats[str(e)] += 1
            except ValueError as e:
                stats[f"ValueError while attempting to generate tensor:\n{traceback.format_exc()}\n"] += 1
                simple_stats[str(e)] += 1
            except OSError as e:
                stats[f"OSError while attempting to generate tensor:\n{traceback.format_exc()}\n"] += 1
                simple_stats[str(e)] += 1
            except RuntimeError as e:
                stats[f"RuntimeError while attempting to generate tensor:\n{traceback.format_exc()}\n"] += 1
                simple_stats[str(e)] += 1
            finally:
                if hd5 is not None:
                    hd5.close()
                _log_first_error(stats, tp)
        stats['epochs'] += 1
        np.random.shuffle(train_paths)
        for k in stats:
            logging.debug(f"{k}: {stats[k]}")
        error_info = '\n\t\t'.join([f'[{error}] - {count}'
                                    for error, count in sorted(simple_stats.items(), key=lambda x: x[1], reverse=True)])
        info_string = '\n\t'.join([
            f"The following errors occurred:\n\t\t{error_info}",
            f"Generator looped & shuffled over {len(train_paths)} tensors.",
            f"{int(stats['Tensors presented']/stats['epochs'])} tensors were presented.",
            f"The cache holds {len(cache)} out of a possible {len(train_paths) * (len(input_maps) + len(output_maps))} tensors and is {100 * cache.average_fill():.0f}% full.",
            f"So far there have been {cache.hits} cache hits.",
            f"{(time.time() - start):.2f} seconds elapsed.",
        ])
        logging.info(f"In true epoch {stats['epochs']}:\n\t{info_string}")
        if stats['Tensors presented'] == 0:
            raise ValueError(f"Completed an epoch but did not find any tensors to yield")


def big_batch_from_minibatch_generator(tensor_maps_in, tensor_maps_out, generator, minibatches, keep_paths=True):
    """Collect minibatches into bigger batches

    Returns a dicts of numpy arrays like the same kind as generator but with more examples.

    Arguments:
        tensor_maps_in: list of TensorMaps that are input names to a model
        tensor_maps_out: list of TensorMaps that are output from a model
        generator: TensorGenerator of minibatches
        minibatches: number of times to call generator and collect a minibatch
        keep_paths: also return the list of tensor files loaded

    Returns:
        A tuple of dicts mapping tensor names to big batches of numpy arrays mapping.
    """     
    input_tensors = {tm.input_name(): [] for tm in tensor_maps_in}
    output_tensors = {tm.output_name(): [] for tm in tensor_maps_out}
    paths = []

    for _ in range(minibatches):
        next_batch = next(generator)
        for key in input_tensors:
            input_tensors[key].extend(np.copy(next_batch[0][key]))
        for key in output_tensors:
            output_tensors[key].extend(np.copy(next_batch[1][key]))
        if keep_paths:
            paths.extend(next_batch[2])
    for key in input_tensors:
        input_tensors[key] = np.array(input_tensors[key])
        logging.info("Input tensor '{}' has shape {}".format(key, input_tensors[key].shape))
    for key in output_tensors:
        output_tensors[key] = np.array(output_tensors[key])
        logging.info("Output tensor '{}' has shape {}".format(key, output_tensors[key].shape))

    if keep_paths:
        return input_tensors, output_tensors, paths
    else:
        return input_tensors, output_tensors


def get_test_train_valid_paths(tensors, valid_ratio, test_ratio, test_modulo):
    """Return 3 disjoint lists of tensor paths.

    The paths are split in training, validation and testing lists
    apportioned according to valid_ratio and test_ratio

    Arguments:
        tensors: directory containing tensors
        valid_ratio: rate of tensors in validation list
        test_ratio: rate of tensors in testing list
        test_modulo: if greater than 1, all sample ids modulo this number will be used for testing regardless of test_ratio and valid_ratio

    Returns:
        A tuple of 3 lists of hd5 tensor file paths
    """     
    test_paths = []
    train_paths = []
    valid_paths = []

    assert valid_ratio > 0 and test_ratio > 0 and valid_ratio+test_ratio < 1.0

    for root, dirs, files in os.walk(tensors):
        for name in files:
            if os.path.splitext(name)[-1].lower() != TENSOR_EXT:
                continue
            dice = np.random.rand()
            if dice < valid_ratio or (test_modulo > 1 and int(os.path.splitext(name)[0]) % test_modulo == 0):
                test_paths.append(os.path.join(root, name))
            elif dice < (valid_ratio+test_ratio):
                valid_paths.append(os.path.join(root, name))
            else:   
                train_paths.append(os.path.join(root, name))

    logging.info(f"Found {len(train_paths)} training, {len(valid_paths)} validation, and {len(test_paths)} testing tensors at: {tensors}")
    if len(train_paths) == 0 or len(valid_paths) == 0 or len(test_paths) == 0:
        raise ValueError(f"Not enough tensors at {tensors}\n")

    return train_paths, valid_paths, test_paths


def get_test_train_valid_paths_split_by_csvs(tensors, balance_csvs, valid_ratio, test_ratio, test_modulo):
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

    test_paths = [[] for _ in range(len(balance_csvs)+1)]
    train_paths = [[] for _ in range(len(balance_csvs)+1)]
    valid_paths = [[] for _ in range(len(balance_csvs)+1)]
    for root, dirs, files in os.walk(tensors):
        for name in files:
            splits = os.path.splitext(name)
            if splits[-1].lower() != TENSOR_EXT:
                continue
                
            group = 0
            sample_id = os.path.basename(splits[0])
            if sample_id in sample2group:
                group = sample2group[sample_id]

            dice = np.random.rand()
            if dice < valid_ratio or (test_modulo > 1 and int(os.path.splitext(name)[0]) % test_modulo == 0):
                test_paths[group].append(os.path.join(root, name))
            elif dice < (valid_ratio+test_ratio):
                valid_paths[group].append(os.path.join(root, name))
            else:
                train_paths[group].append(os.path.join(root, name))

    for i in range(len(train_paths)):
        if len(train_paths[i]) == 0 or len(valid_paths[i]) == 0 or len(test_paths[i]) == 0:
            my_error = f"Not enough tensors at {tensors}\nGot {len(train_paths[i])} train {len(valid_paths[i])} valid and {len(test_paths[i])} test."
            raise ValueError(my_error)
        if i == 0:
            logging.info(f"Found {len(train_paths[i])} train {len(valid_paths[i])} valid and {len(test_paths[i])} test tensors outside the CSVs.")
        else:
            logging.info(f"CSV:{balance_csvs[i-1]}\nhas: {len(train_paths[i])} train, {len(valid_paths[i])} valid, {len(test_paths[i])} test tensors.")
    
    return train_paths, valid_paths, test_paths


@contextmanager
def test_train_valid_tensor_generators(tensor_maps_in: List[TensorMap],
                                       tensor_maps_out: List[TensorMap],
                                       tensors: str,
                                       batch_size: int,
                                       valid_ratio: float,
                                       test_ratio: float,
                                       test_modulo: int,
                                       num_workers: int,
                                       cache_size: float,
                                       balance_csvs: List[str],
                                       keep_paths: bool = False,
                                       keep_paths_test: bool = True,
                                       mixup_alpha: float = -1.0,
                                       **kwargs) -> Tuple[TensorGenerator, TensorGenerator, TensorGenerator]:
    """ Get 3 tensor generator functions for training, validation and testing data.

    :param maps_in: list of TensorMaps that are input names to a model
    :param maps_out: list of TensorMaps that are output from a model
    :param tensors: directory containing tensors
    :param batch_size: number of examples in each mini-batch
    :param valid_ratio: rate of tensors to use for validation
    :param test_ratio: rate of tensors to use for testing
    :param test_modulo: if greater than 1, all sample ids modulo this number will be used for testing regardless of test_ratio and valid_ratio
    :param balance_csvs: if not empty, generator will provide batches balanced amongst the Sample ID in these CSVs.
    :param keep_paths: also return the list of tensor files loaded for training and validation tensors
    :param keep_paths_test:  also return the list of tensor files loaded for testing tensors
    :return: A tuple of three generators. Each yields a Tuple of dictionaries of input and output numpy arrays for training, validation and testing.
    """
    generate_train, generate_valid, generate_test = None, None, None
    try:
        if len(balance_csvs) > 0:
            train_paths, valid_paths, test_paths = get_test_train_valid_paths_split_by_csvs(tensors, balance_csvs, valid_ratio, test_ratio, test_modulo)
            weights = [1.0/(len(balance_csvs)+1) for _ in range(len(balance_csvs)+1)]
        else:
            train_paths, valid_paths, test_paths = get_test_train_valid_paths(tensors, valid_ratio, test_ratio, test_modulo)
            weights = None
        generate_train = TensorGenerator(batch_size, tensor_maps_in, tensor_maps_out, train_paths, num_workers, cache_size, weights, keep_paths, mixup_alpha, name='train_worker')
        generate_valid = TensorGenerator(batch_size, tensor_maps_in, tensor_maps_out, valid_paths, num_workers, cache_size, None, keep_paths, name='validation_worker')
        generate_test = TensorGenerator(batch_size, tensor_maps_in, tensor_maps_out, test_paths, num_workers, cache_size, None, keep_paths or keep_paths_test, name='test_worker')
        yield generate_train, generate_valid, generate_test
    finally:
        for generator in (generate_train, generate_valid, generate_test):
            if generator is not None:
                generator.kill()


def _log_first_error(stats: Counter, tensor_path: str):
    for k in stats:
        if 'Error' in k and stats[k] == 1:
            stats[k] += 1  # Increment so we only see these messages once
            logging.debug(f"At tensor path: {tensor_path}")
            logging.debug(f"Got first error: {k}")


def _mixup_batch(in_batch: Dict[str, np.ndarray], out_batch: Dict[str, np.ndarray], alpha: float = 1.0, permute_first: bool = False):
    for k in in_batch:
        full_batch = in_batch[k].shape[0]
        half_batch = full_batch // 2
        break

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

    return mixed_ins, mixed_outs
