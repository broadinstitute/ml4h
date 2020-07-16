# On-the-fly data generation of tensors for training or prediction

# Python 2/3 friendly
from __future__ import print_function

# Imports: standard library
# Imports
import os
import csv
import math
import time
import logging
import traceback
from typing import Any, Set, Dict, List, Tuple, Union, Callable, Iterator, Optional
from itertools import chain
from collections import Counter, defaultdict
from multiprocessing import Event, Queue, Barrier, Process

# Imports: third party
import h5py
import numpy as np
import pandas as pd

# Imports: first party
from ml4cvd.defines import TENSOR_EXT
from ml4cvd.TensorMap import TensorMap

np.set_printoptions(threshold=np.inf)


DEFAULT_VALID_RATIO = 0.2
DEFAULT_TEST_RATIO = 0.1

TENSOR_GENERATOR_TIMEOUT = 64
TENSOR_GENERATOR_MAX_Q_SIZE = 32

# TensorGenerator batch indices
BATCH_INPUT_INDEX, BATCH_OUTPUT_INDEX, BATCH_SAMPLE_WEIGHTS_INDEX, BATCH_PATHS_INDEX = (
    0,
    1,
    2,
    3,
)

Path = str
PathIterator = Iterator[Path]
Batch = Dict[Path, np.ndarray]
BatchFunction = Callable[[Batch, Batch, bool, List[Path], "kwargs"], Any]


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
            raise ValueError("Weights must be the same length as paths.")

    def __next__(self) -> str:
        return np.random.choice(np.random.choice(self.paths, self.weights))


class TensorGenerator:
    def __init__(
        self,
        batch_size: int,
        input_maps: List[TensorMap],
        output_maps: List[TensorMap],
        paths: Union[List[str], List[List[str]]],
        num_workers: int,
        cache_size,  # no cache is used but include to match type signature of v1
        weights: List[float] = None,
        keep_paths: bool = False,
        mixup: float = 0.0,
        name: str = "worker",
        siamese: bool = False,
        augment: bool = False,
        sample_weight: TensorMap = None,
    ):
        self.batch_size = batch_size
        self.input_maps = input_maps
        self.output_maps = output_maps
        self.paths = paths
        self.weights = weights
        self.keep_paths = keep_paths
        self.mixup = mixup
        self.name = name
        self.siamese = siamese
        self.augment = augment
        self.sample_weight = sample_weight

        self.run_on_main_thread = num_workers == 0
        num_workers = num_workers or 1
        self.num_workers = num_workers
        self.started = False
        self.worker_q = None
        self.worker_barrier = None
        self.worker_processes = []
        self.worker_instances = []
        self.stats = Counter()
        self.true_epoch_stats = self.stats

        self.true_epoch_len = len(paths)
        if weights is None:
            worker_paths = np.array_split(paths, num_workers)
            self.worker_true_epoch_lens = list(map(len, worker_paths))
            self.worker_path_iters = [_ShufflePaths(p) for p in worker_paths]
        else:
            # split each path list into paths for each worker.
            # E.g. for two workers: [[p1, p2], [p3, p4, p5]] -> [[[p1], [p2]], [[p3, p4], [p5]]
            split_paths = [np.array_split(a, num_workers) for a in paths]
            # Next, each list of paths gets given to each worker. E.g. [[[p1], [p3, p4]], [[p2], [p5]]]
            worker_paths = np.swapaxes(split_paths, 0, 1)
            self.worker_true_epoch_lens = [max(map(len, p)) for p in worker_paths]
            self.worker_path_iters = [_WeightedPaths(p, weights) for p in worker_paths]

        self.batch_function_kwargs = {}
        if mixup > 0:
            self.batch_function = _mixup_batch
            self.batch_size *= 2
            self.batch_function_kwargs = {"alpha": mixup}
        elif siamese:
            self.batch_function = _make_batch_siamese
        elif sample_weight:
            self.input_maps = input_maps[:] + [sample_weight]
            self.batch_function = _weighted_batch
            self.batch_function_kwargs = {"sample_weight": sample_weight}
        else:
            self.batch_function = _identity_batch

    def __next__(self):
        if not self.started:
            self.init_workers()

        in_batch = {
            tm.input_name(): np.zeros((self.batch_size,) + tm.static_shape())
            for tm in self.input_maps
        }
        out_batch = {
            tm.output_name(): np.zeros((self.batch_size,) + tm.static_shape())
            for tm in self.output_maps
        }
        paths = []

        self.stats["batch_count"] += 1
        self.stats["batch_total"] += 1
        i = 0  # i changes in the body of the loop, cannot simply use range()
        while i < self.batch_size:
            self.stats["sample_count"] += 1
            self.stats["sample_total"] += 1
            if self.run_on_main_thread:
                tensors = next(self.worker_instances[0])
            else:
                tensors = self.worker_q.get()
            _collect_tensor_stats(self, tensors)
            in_tensors, out_tensors, path, error = tensors
            if error is not None:
                self.stats["samples_failed"] += 1
                i -= 1  # rollback batch index, only use tensors that succeed
            else:
                for tm, in_tensor in in_tensors.items():
                    in_batch[tm][i] = in_tensor
                for tm, out_tensor in out_tensors.items():
                    out_batch[tm][i] = out_tensor
                paths.append(path)
                self.stats["samples_succeeded"] += 1

            if self.reached_true_epoch():
                if not self.run_on_main_thread:
                    self.worker_barrier.wait()
                self.stats["true_epochs"] += 1

                logging.info(
                    f"{get_stats_string(self)}{get_verbose_stats_string({self.name: self})}",
                )

                # reset stats
                self.true_epoch_stats = self.stats
                self.stats = Counter()
                self.stats["true_epochs"] = self.true_epoch_stats["true_epochs"]
                self.stats["batch_total"] = self.true_epoch_stats["batch_total"]
                self.stats["sample_total"] = self.true_epoch_stats["sample_total"]

            i += 1

        batch = self.batch_function(
            in_batch, out_batch, self.keep_paths, paths, **self.batch_function_kwargs
        )
        return batch

    def reached_true_epoch(self):
        return self.stats["sample_count"] == self.true_epoch_len

    def __iter__(self):
        return self

    def init_workers(self):
        self.worker_q = Queue()
        self.worker_barrier = Barrier(self.num_workers + 1)
        self.started = True
        for i, (worker_path_iter, worker_true_epoch_len) in enumerate(
            zip(self.worker_path_iters, self.worker_true_epoch_lens),
        ):
            name = f"{self.name}_{i}"
            worker_instance = TensorGeneratorWorker(
                name,
                worker_path_iter,
                worker_true_epoch_len,
                self.input_maps,
                self.output_maps,
                self.augment,
                self.worker_q,
                self.worker_barrier,
            )
            self.worker_instances.append(worker_instance)
            if not self.run_on_main_thread:
                worker_process = Process(
                    target=worker_instance.run_concurrent, name=name, args=(),
                )
                worker_process.start()
                self.worker_processes.append(worker_process)
        if not self.run_on_main_thread:
            logging.info(f"Started {self.num_workers} {self.name}s")

    def kill_workers(self):
        if self.started:
            for worker_process in self.worker_processes:
                worker_process.terminate()
            self.started = False
            self.worker_q = None
            self.worker_barrier = None
            self.worker_instances = []
            self.worker_processes = []
            if not self.run_on_main_thread:
                logging.info(f"Stopped {self.num_workers} {self.name}s")

    def __del__(self):
        self.kill_workers()


class TensorGeneratorWorker:
    def __init__(
        self,
        name: str,
        path_iter: PathIterator,
        true_epoch_len: int,
        input_maps: List[TensorMap],
        output_maps: List[TensorMap],
        augment: bool,
        q: Queue,
        barrier: Barrier,
    ):
        self.name = name
        self.path_iter = path_iter
        self.true_epoch_len = true_epoch_len
        self.input_maps = input_maps
        self.output_maps = output_maps
        self.augment = augment
        self.q = q
        self.barrier = barrier
        self.count = 0

    def __next__(
        self,
    ) -> Tuple[
        Optional[Dict[str, np.ndarray]],
        Optional[Dict[str, np.ndarray]],
        str,
        Optional[Exception],
    ]:
        self.count += 1

        path = next(self.path_iter)
        hd5 = None
        try:
            hd5 = h5py.File(path, "r")
            dependents = {}
            in_tensors = {
                tm.input_name(): self.get_tensor(tm, hd5, dependents, input=True)
                for tm in self.input_maps
            }
            out_tensors = {
                tm.output_name(): self.get_tensor(tm, hd5, dependents, input=False)
                for tm in self.output_maps
            }
            return in_tensors, out_tensors, path, None
        except (IndexError, KeyError, ValueError, OSError, RuntimeError) as e:
            return None, None, path, e
        finally:
            if hd5 is not None:
                hd5.close()

    def get_tensor(
        self, tm: TensorMap, hd5: h5py.File, dependents: Dict, input: bool,
    ) -> np.ndarray:
        name = tm.input_name() if input else tm.output_name()
        if name in dependents:
            return dependents[name]
        return tm.postprocess_tensor(
            tm.tensor_from_file(tm, hd5, dependents), self.augment, hd5,
        )

    def run_concurrent(self):
        while True:
            tensors = next(self)
            # before putting the first tensor of the next true epoch in the queue
            # wait for all the tensors in the previous true epoch to be consumed
            # allows loading of the first tensor without enqueueing
            if self.count != 1 and self.count % self.true_epoch_len == 1:
                self.barrier.wait()
            self.q.put(tensors)


def _collect_tensor_stats(
    generator: TensorGenerator,
    tensors: Tuple[
        Optional[Dict[str, np.ndarray]],
        Optional[Dict[str, np.ndarray]],
        str,
        Optional[Exception],
    ],
) -> None:
    stats = generator.stats
    in_tensors, out_tensors, path, error = tensors
    if error is not None:
        error_name = type(error).__name__
        stats[f"error_{error_name}"] += 1
    else:
        for tms, tensors, is_input in [
            (generator.input_maps, in_tensors, True),
            (generator.output_maps, out_tensors, False),
        ]:
            for tm in tms:
                stats[f"{tm.name}_n"] += 1
                if tm.axes() == 1:
                    tensor = tensors[tm.input_name() if is_input else tm.output_name()]
                    if tm.is_categorical():
                        stats[f"{tm.name}_index_{np.argmax(tensor):.0f}"] += 1
                    elif tm.is_continuous():
                        value = tensor[0]
                        min_key = f"{tm.name}_min"
                        max_key = f"{tm.name}_max"
                        if min_key not in stats or value < stats[min_key]:
                            stats[min_key] = value
                        if max_key not in stats or value > stats[max_key]:
                            stats[max_key] = value
                        stats[f"{tm.name}_sum"] += value
                        stats[f"{tm.name}_squared_sum"] += value ** 2


def get_stats_string(generator: TensorGenerator) -> str:
    stats = generator.true_epoch_stats
    # fmt: off
    return (
        f"\n"
        f"------------------- {generator.name} completed true epoch {stats['true_epochs']} -------------------\n"
        f"\tGenerator shuffled {stats['sample_count']} samples.\n"
        f"\t{stats['samples_succeeded']} samples successfully yielded tensors.\n"
        f"\t{stats['samples_failed']} samples failed to yield tensors.\n"
    )
    # fmt: on


def get_verbose_stats_string(generators: Dict[str, TensorGenerator]) -> str:
    if len(generators) == 1:
        generator = list(generators.values())[0]
        dataframes = _get_stats_as_dataframes(
            generator.true_epoch_stats, generator.input_maps, generator.output_maps,
        )
    else:
        dataframes = _get_stats_as_dataframes_from_multiple_generators(generators)
    continuous_tm_df, categorical_tm_df, other_tm_df = dataframes

    continuous_tm_string = (
        f">>>>>>>>>> Continuous Tensor Maps\n{continuous_tm_df}"
        if len(continuous_tm_df) != 0
        else ""
    )

    categorical_tm_strings = []
    for tm in categorical_tm_df.index.get_level_values("TensorMap").drop_duplicates():
        tm_df = categorical_tm_df.loc[tm]
        categorical_tm_strings.append(
            f">>>>>>>>>> Categorical Tensor Map: [{tm}]\n{tm_df}",
        )

    other_tm_string = (
        f">>>>>>>>>> Other Tensor Maps\n{other_tm_df}" if len(other_tm_df) != 0 else ""
    )

    tensor_stats_string = "\n\n".join(
        [
            s
            for s in [continuous_tm_string] + categorical_tm_strings + [other_tm_string]
            if s != ""
        ],
    )

    return f"\n{tensor_stats_string}\n"


def _get_stats_as_dataframes(
    stats: Counter, input_maps: List[TensorMap], output_maps: List[TensorMap],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    continuous_tmaps = []
    categorical_tmaps = []
    other_tmaps = []
    for tm in input_maps + output_maps:
        if tm.axes() == 1 and tm.is_continuous():
            continuous_tmaps.append(tm)
        elif tm.axes() == 1 and tm.is_categorical():
            categorical_tmaps.append(tm)
        else:
            other_tmaps.append(tm)

    _stats = defaultdict(list)
    for tm in continuous_tmaps:
        count = stats[f"{tm.name}_n"]
        mean = stats[f"{tm.name}_sum"] / count
        std = np.sqrt((stats[f"{tm.name}_squared_sum"] / count) - (mean ** 2))
        _stats["count"].append(f"{count:.0f}")
        _stats["mean"].append(f"{mean:.2f}")
        _stats["std"].append(f"{std:.2f}")
        _stats["min"].append(f"{stats[f'{tm.name}_min']:.2f}")
        _stats["max"].append(f"{stats[f'{tm.name}_max']:.2f}")
    continuous_tm_df = pd.DataFrame(_stats, index=[tm.name for tm in continuous_tmaps])
    continuous_tm_df.index.name = "TensorMap"

    _stats = defaultdict(list)
    for tm in categorical_tmaps:
        total = stats[f"{tm.name}_n"]
        for channel, index in tm.channel_map.items():
            count = stats[f"{tm.name}_index_{index}"]
            _stats["count"].append(f"{count:.0f}")
            _stats["percent"].append(f"{count / total * 100:.2f}")
            _stats["TensorMap"].append(tm.name)
            _stats["Label"].append(channel)
        _stats["count"].append(f"{total:.0f}")
        _stats["percent"].append("100.00")
        _stats["TensorMap"].append(tm.name)
        _stats["Label"].append("total")
    categorical_tm_df = pd.DataFrame(
        {key: val for key, val in _stats.items() if key in {"count", "percent"}},
        index=pd.MultiIndex.from_tuples(
            zip(_stats["TensorMap"], _stats["Label"]), names=["TensorMap", "Label"],
        ),
    )
    categorical_tm_df.index.name = "TensorMap"

    other_tm_df = pd.DataFrame(
        {"count": [f"{stats[f'{tm.name}_n']:.0f}" for tm in other_tmaps]},
        index=[tm.name for tm in other_tmaps],
    )
    other_tm_df.index.name = "TensorMap"

    return continuous_tm_df, categorical_tm_df, other_tm_df


def _get_stats_as_dataframes_from_multiple_generators(
    generators: Dict[str, TensorGenerator],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    split_to_dataframes = {
        split: _get_stats_as_dataframes(
            stats=generator.true_epoch_stats,
            input_maps=generator.input_maps,
            output_maps=generator.output_maps,
        )
        for split, generator in generators.items()
    }

    def combine_split_dataframes(split_to_dataframe, index=["TensorMap"]):
        df = (
            pd.concat(split_to_dataframe, names=["Split"])
            .reorder_levels(index + ["Split"])
            .reset_index()
        )
        df["Split"] = pd.Categorical(df["Split"], split_to_dataframe.keys())
        if "Label" in index:
            labels = df["Label"].drop_duplicates()
            df["Label"] = pd.Categorical(df["Label"], labels)
        df = df.set_index(index + ["Split"]).sort_index()
        return df

    continuous_tm_df = combine_split_dataframes(
        {
            split: continuous_df
            for split, (continuous_df, _, _) in split_to_dataframes.items()
        },
    )
    categorical_tm_df = combine_split_dataframes(
        {
            split: categorical_df
            for split, (_, categorical_df, _) in split_to_dataframes.items()
        },
        ["TensorMap", "Label"],
    )
    other_tm_df = combine_split_dataframes(
        {split: other_df for split, (_, _, other_df) in split_to_dataframes.items()},
    )

    return continuous_tm_df, categorical_tm_df, other_tm_df


def big_batch_from_minibatch_generator(generator: TensorGenerator, minibatches: int):
    """Collect minibatches into bigger batches

    Returns a dicts of numpy arrays like the same kind as generator but with more examples.

    Arguments:
        generator: TensorGenerator of minibatches
        minibatches: number of times to call generator and collect a minibatch

    Returns:
        A tuple of dicts mapping tensor names to big batches of numpy arrays mapping.
    """
    first_batch = next(generator)
    saved_tensors = {}
    batch_size = None
    for key, batch_array in chain(
        first_batch[BATCH_INPUT_INDEX].items(), first_batch[BATCH_OUTPUT_INDEX].items(),
    ):
        shape = (batch_array.shape[0] * minibatches,) + batch_array.shape[1:]
        saved_tensors[key] = np.zeros(shape)
        batch_size = batch_array.shape[0]
        saved_tensors[key][:batch_size] = batch_array

    keep_paths = generator.keep_paths
    if keep_paths:
        paths = first_batch[BATCH_PATHS_INDEX]

    input_tensors, output_tensors = (
        list(first_batch[BATCH_INPUT_INDEX]),
        list(first_batch[BATCH_OUTPUT_INDEX]),
    )
    for i in range(1, minibatches):
        logging.debug(f"big_batch_from_minibatch {100 * i / minibatches:.2f}% done.")
        next_batch = next(generator)
        s, t = i * batch_size, (i + 1) * batch_size
        for key in input_tensors:
            saved_tensors[key][s:t] = next_batch[BATCH_INPUT_INDEX][key]
        for key in output_tensors:
            saved_tensors[key][s:t] = next_batch[BATCH_OUTPUT_INDEX][key]
        if keep_paths:
            paths.extend(next_batch[BATCH_PATHS_INDEX])

    for key, array in saved_tensors.items():
        logging.info(
            f"Made a big batch of tensors with key:{key} and shape:{array.shape}.",
        )
    inputs = {key: saved_tensors[key] for key in input_tensors}
    outputs = {key: saved_tensors[key] for key in output_tensors}
    if keep_paths:
        return inputs, outputs, paths
    else:
        return inputs, outputs


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
        raise ValueError(
            "ratios do not sum to 1, train/valid/test/discard ="
            f" {train_ratio}/{valid_ratio}/{test_ratio}/{discard_ratio}",
        )
    logging.debug(
        "train/valid/test/discard ratios:"
        f" {train_ratio}/{valid_ratio}/{test_ratio}/{discard_ratio}",
    )

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
            f"{sample_csv} has more than one potential column for MRNs. Inferring most"
            " likely column name, but recommend explicitly setting MRN column name.",
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

    (
        train_ratio,
        valid_ratio,
        test_ratio,
        discard_ratio,
    ) = _get_train_valid_test_discard_ratios(
        valid_ratio=valid_ratio,
        test_ratio=test_ratio,
        train_csv=train_csv,
        valid_csv=valid_csv,
        test_csv=test_csv,
    )

    choices = {
        "train": (train_paths, train_ratio),
        "valid": (valid_paths, valid_ratio),
        "test": (test_paths, test_ratio),
        "discard": (discard_paths, discard_ratio),
    }

    # parse csv's to disjoint sets, None if csv was None
    sample_set = _sample_csv_to_set(sample_csv)

    train_set = _sample_csv_to_set(train_csv)
    valid_set = _sample_csv_to_set(valid_csv)
    test_set = _sample_csv_to_set(test_csv)

    if (
        train_set is not None
        and valid_set is not None
        and not train_set.isdisjoint(valid_set)
    ):
        raise ValueError("train and validation samples overlap")
    if (
        train_set is not None
        and test_set is not None
        and not train_set.isdisjoint(test_set)
    ):
        raise ValueError("train and test samples overlap")
    if (
        valid_set is not None
        and test_set is not None
        and not valid_set.isdisjoint(test_set)
    ):
        raise ValueError("validation and test samples overlap")

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
                choice = np.random.choice(
                    [k for k in choices], p=[choices[k][1] for k in choices],
                )
                choices[choice][0].append(path)

    logging.info(
        f"Found {len(train_paths)} train, {len(valid_paths)} validation, and"
        f" {len(test_paths)} testing tensors at: {tensors}",
    )
    logging.debug(f"Discarded {len(discard_paths)} tensors due to given ratios")
    if len(train_paths) == 0 or len(valid_paths) == 0 or len(test_paths) == 0:
        raise ValueError(
            f"Not enough tensors at {tensors}\n"
            f"Found {len(train_paths)} training,"
            f" {len(valid_paths)} validation, and"
            f" {len(test_paths)} testing tensors\n"
            f"Discarded {len(discard_paths)} tensors",
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
        lol = list(csv.reader(open(b_csv, "r"), delimiter=","))
        logging.info(f"Class Balance CSV Header: {list(enumerate(lol[0]))}")

        for row in lol[1:]:
            sample_id = row[0]
            sample2group[sample_id] = i + 1  # group 0 means background class
            stats["group_" + str(i + 1)] += 1
    logging.info(f"Balancing with CSVs of Sample IDs stats: {stats}")

    train_paths = [[] for _ in range(len(balance_csvs) + 1)]
    valid_paths = [[] for _ in range(len(balance_csvs) + 1)]
    test_paths = [[] for _ in range(len(balance_csvs) + 1)]

    _train, _valid, _test = get_train_valid_test_paths(
        tensors=tensors,
        sample_csv=sample_csv,
        valid_ratio=valid_ratio,
        test_ratio=test_ratio,
        train_csv=train_csv,
        valid_csv=valid_csv,
        test_csv=test_csv,
    )

    for paths, split_list in [
        (_train, train_paths),
        (_valid, valid_paths),
        (_test, test_paths),
    ]:
        for path in paths:
            split = os.path.splitext(os.path.basename(path))
            sample_id = split[0]

            group = 0
            if sample_id in sample2group:
                group = sample2group[sample_id]
            split_list[group].append(path)

    for i in range(len(train_paths)):
        if (
            len(train_paths[i]) == 0
            or len(valid_paths[i]) == 0
            or len(test_paths[i]) == 0
        ):
            my_error = (
                f"Not enough tensors at {tensors}\nGot {len(train_paths[i])} train"
                f" {len(valid_paths[i])} valid and {len(test_paths[i])} test."
            )
            raise ValueError(my_error)
        if i == 0:
            logging.info(
                f"Found {len(train_paths[i])} train {len(valid_paths[i])} valid and"
                f" {len(test_paths[i])} test tensors outside the CSVs.",
            )
        else:
            logging.info(
                f"CSV:{balance_csvs[i-1]}\nhas: {len(train_paths[i])} train,"
                f" {len(valid_paths[i])} valid, {len(test_paths[i])} test tensors.",
            )
    return train_paths, valid_paths, test_paths


def test_train_valid_tensor_generators(
    tensor_maps_in: List[TensorMap],
    tensor_maps_out: List[TensorMap],
    tensors: str,
    batch_size: int,
    num_workers: int,
    training_steps: int,
    validation_steps: int,
    cache_size: float,
    balance_csvs: List[str],
    keep_paths: bool = False,
    keep_paths_test: bool = True,
    mixup_alpha: float = -1.0,
    sample_csv: str = None,
    valid_ratio: float = None,
    test_ratio: float = None,
    train_csv: str = None,
    valid_csv: str = None,
    test_csv: str = None,
    siamese: bool = False,
    sample_weight: TensorMap = None,
    **kwargs,
) -> Tuple[TensorGenerator, TensorGenerator, TensorGenerator]:
    """Get 3 tensor generator functions for training, validation and testing data.

    :param tensor_maps_in: list of TensorMaps that are input names to a model
    :param tensor_maps_out: list of TensorMaps that are output from a model
    :param tensors: directory containing tensors
    :param batch_size: number of examples in each mini-batch
    :param num_workers: number of processes spun off for training and testing. Validation uses half as many workers
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
    :param sample_weight: TensorMap that outputs a sample weight for the other tensors
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
        weights = [1.0 / (len(balance_csvs) + 1) for _ in range(len(balance_csvs) + 1)]
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

    _TensorGenerator = TensorGenerator

    generate_train = _TensorGenerator(
        batch_size,
        tensor_maps_in,
        tensor_maps_out,
        train_paths,
        num_workers,
        cache_size,
        weights,
        keep_paths,
        mixup_alpha,
        name="train_worker",
        siamese=siamese,
        augment=True,
        sample_weight=sample_weight,
    )
    generate_valid = _TensorGenerator(
        batch_size,
        tensor_maps_in,
        tensor_maps_out,
        valid_paths,
        num_workers,
        cache_size,
        weights,
        keep_paths,
        name="validation_worker",
        siamese=siamese,
        augment=False,
    )
    generate_test = _TensorGenerator(
        batch_size,
        tensor_maps_in,
        tensor_maps_out,
        test_paths,
        num_workers,
        0,
        weights,
        keep_paths or keep_paths_test,
        name="test_worker",
        siamese=siamese,
        augment=False,
    )
    return generate_train, generate_valid, generate_test


def _identity_batch(
    in_batch: Batch, out_batch: Batch, return_paths: bool, paths: List[Path],
):
    sample_weights = [None] * len(out_batch)
    return (
        (in_batch, out_batch, sample_weights, paths)
        if return_paths
        else (in_batch, out_batch, sample_weights)
    )


def _mixup_batch(
    in_batch: Batch,
    out_batch: Batch,
    return_paths: bool,
    paths: List[Path],
    alpha: float = 1.0,
    permute_first: bool = False,
):
    full_batch = in_batch.values().__iter__().__next__().shape[0]
    half_batch = full_batch // 2

    if permute_first:
        permuted = np.random.permutation(full_batch)
        for k in in_batch:
            in_batch[k] = in_batch[k][permuted, ...]
        for k in out_batch:
            out_batch[k] = out_batch[k][permuted, ...]

    mixed_ins = {k: np.zeros((half_batch,) + in_batch[k].shape[1:]) for k in in_batch}
    mixed_outs = {
        k: np.zeros((half_batch,) + out_batch[k].shape[1:]) for k in out_batch
    }
    for i in range(half_batch):
        weight0 = np.random.beta(alpha, alpha)
        weight1 = 1 - weight0
        for k in in_batch:
            mixed_ins[k][i] = (in_batch[k][i, ...] * weight0) + (
                in_batch[k][half_batch + i, ...] * weight1
            )
        for k in out_batch:
            mixed_outs[k][i] = (out_batch[k][i, ...] * weight0) + (
                out_batch[k][half_batch + i, ...] * weight1
            )

    return _identity_batch(mixed_ins, mixed_outs, return_paths, paths[:half_batch])


def _make_batch_siamese(
    in_batch: Batch, out_batch: Batch, return_paths: bool, paths: List[Path],
):
    full_batch = in_batch.values().__iter__().__next__().shape[0]
    half_batch = full_batch // 2

    siamese_in = {
        k + "_left": np.zeros((half_batch,) + in_batch[k].shape[1:]) for k in in_batch
    }
    siamese_in.update(
        {
            k + "_right": np.zeros((half_batch,) + in_batch[k].shape[1:])
            for k in in_batch
        },
    )
    siamese_out = {"output_siamese": np.zeros((half_batch, 1))}

    for i in range(half_batch):
        for k in in_batch:
            siamese_in[k + "_left"][i] = in_batch[k][i, ...]
            siamese_in[k + "_right"][i] = in_batch[k][half_batch + i, ...]
        random_task_key = np.random.choice(list(out_batch.keys()))
        siamese_out["output_siamese"][i] = (
            0
            if np.array_equal(
                out_batch[random_task_key][i],
                out_batch[random_task_key][i + half_batch],
            )
            else 1
        )

    return _identity_batch(siamese_in, siamese_out, return_paths, paths)


def _weighted_batch(
    in_batch: Batch,
    out_batch: Batch,
    return_paths: bool,
    paths: List[Path],
    sample_weight: TensorMap,
):
    sample_weights = [in_batch.pop(sample_weight.input_name()).flatten()] * len(
        out_batch,
    )
    return (
        (in_batch, out_batch, sample_weights, paths)
        if return_paths
        else (in_batch, out_batch, sample_weights)
    )
