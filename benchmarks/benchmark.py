import os
import sys
import time
# import zarr
import shutil
import datetime
import pandas as pd
import seaborn as sns
import tensorflow as tf
from itertools import product
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from typing import List, Generator
from abc import ABC, abstractmethod
from ml4h.defines import StorageType
from contextlib import contextmanager
from multiprocessing import cpu_count

from data import build_tensor_maps, build_hd5s_ukbb, get_hd5_paths, DataDescription, SYNTHETIC_DATA_PATH
# from data import ZarrGenerator, build_zarr, build_zarr_giant_array
# from data import tf_data_zarr, ZarrSequence, _ZarrMap
from data import build_hangar_repo, make_hangar_tf_dset
from data import make_hangar_torch_dset


DELTA_COL = 'step_delta_seconds'
WORKER_COL = 'num_workers'
BATCH_SIZE_COL = 'batch_size'
NAME_COL = 'name'


class GeneratorFactory(ABC):
    is_setup = False

    @abstractmethod
    def setup(self, num_samples: int, data_descriptions: List[DataDescription]):
        pass

    @abstractmethod
    @contextmanager
    def __call__(self, batch_size: int, num_workers: int) -> Generator:
        pass

    @abstractmethod
    def get_name(self) -> str:
        pass


class TensorGeneratorFactory(GeneratorFactory):

    def __init__(self, compression: str):
        super().__init__()
        self.compression = compression
        self.tmaps = None
        self.paths = None

    def get_name(self) -> str:
        return f'TensorGenerator_{self.compression}'

    def setup(self, num_samples: int, data_descriptions: List[DataDescription]):
        build_hd5s_ukbb(data_descriptions, num_samples, overwrite=True, compression=self.compression)
        self.paths = get_hd5_paths(True, num_samples)
        self.tmaps = build_tensor_maps(data_descriptions)

    @contextmanager
    def __call__(self, batch_size: int, num_workers: int) -> Generator:
        from ml4h.tensor_generators import TensorGenerator
        gen = TensorGenerator(
            batch_size=batch_size, num_workers=num_workers,
            input_maps=self.tmaps, output_maps=[],
            cache_size=0, paths=self.paths,
        )
        yield gen
        gen.kill_workers()
        del gen


class ZarrFactory(GeneratorFactory):
    def __init__(self, store_type, store_name: str, giant: bool):
        self.num_samples = None
        self.data_descriptions = None
        self.store_name = store_name
        self.store_type = store_type
        self.giant = giant

    def get_name(self) -> str:
        return f'zarr_{self.store_name}{"_giant" if self.giant else ""}'

    def setup(self, num_samples: int, data_descriptions: List[DataDescription]):
        if self.giant:
            build_zarr_giant_array(data_descriptions, num_samples, self.store_type)
        else:
            build_zarr(data_descriptions, num_samples, self.store_type)
        self.num_samples = num_samples
        self.data_descriptions = data_descriptions

    @contextmanager
    def __call__(self, batch_size: int, num_workers: int) -> Generator:
        gen = ZarrGenerator(
            self.num_samples,
            self.data_descriptions,
            num_workers,
            batch_size,
            self.giant,
            self.store_type,
        )
        yield gen
        gen.kill()


class TFFactory(GeneratorFactory):
    def __init__(self, store_type, store_name: str):
        self.num_samples = None
        self.data_descriptions = None
        self.store_name = store_name
        self.store_type = store_type

    def get_name(self) -> str:
        return f'tf_data_zarr_{self.store_name}'

    def setup(self, num_samples: int, data_descriptions: List[DataDescription]):
        build_zarr(data_descriptions, num_samples, self.store_type)
        self.num_samples = num_samples
        self.data_descriptions = data_descriptions

    @contextmanager
    def __call__(self, batch_size: int, num_workers: int) -> Generator:
        yield tf_data_zarr(self.num_samples, self.data_descriptions, num_workers, batch_size, self.store_type)


class ZarrSequenceFactory(GeneratorFactory):
    def __init__(self, store_type, store_name: str):
        self.num_samples = None
        self.data_descriptions = None
        self.store_name = store_name
        self.store_type = store_type

    def get_name(self) -> str:
        return f'keras_sequence_zarr_{self.store_name}'

    def setup(self, num_samples: int, data_descriptions: List[DataDescription]):
        build_zarr_giant_array(data_descriptions, num_samples, self.store_type)
        self.num_samples = num_samples
        self.data_descriptions = data_descriptions

    @contextmanager
    def __call__(self, batch_size: int, num_workers: int) -> Generator:
        sequence = ZarrSequence(self.num_samples, self.data_descriptions, batch_size, self.store_type)
        enq = tf.keras.utils.OrderedEnqueuer(sequence, use_multiprocessing=True, shuffle=True)
        enq.start(workers=num_workers)
        gen = enq.get()
        yield gen
        enq.stop()


class TFMultiMapFactory(GeneratorFactory):
    def __init__(self, store_type, store_name: str):
        self.num_samples = None
        self.data_descriptions = None
        self.store_name = store_name
        self.store_type = store_type

    def get_name(self) -> str:
        return f'pyfunc_tf_map_zarr_{self.store_name}'

    def setup(self, num_samples: int, data_descriptions: List[DataDescription]):
        build_zarr_giant_array(data_descriptions, num_samples, self.store_type)
        self.num_samples = num_samples
        self.data_descriptions = data_descriptions

    @contextmanager
    def __call__(self, batch_size: int, num_workers: int) -> Generator:
        zmap = _ZarrMap(self.num_samples, self.data_descriptions, self.store_type)
        dset = zmap.build_dataset(num_workers, batch_size)
        yield dset
        zmap.close()


class HangarFactory(GeneratorFactory):
    def __init__(self, torch: bool):
        self.torch = torch

    def get_name(self) -> str:
        return f'hangar_{"torch" if self.torch else "tf_dataset"}'

    def setup(self, num_samples: int, data_descriptions: List[DataDescription]):
        build_hangar_repo(num_samples, data_descriptions)
        self.data_descriptions = data_descriptions

    @contextmanager
    def __call__(self, batch_size: int, num_workers: int) -> Generator:
        if self.torch:
            with make_hangar_torch_dset(self.data_descriptions, batch_size, num_workers) as dset:
                yield dset
        else:
            with make_hangar_tf_dset(self.data_descriptions, batch_size) as dset:
                yield dset


FACTORIES = [
    # ZarrFactory(zarr.DirectoryStore, 'directory_store', True),
    # ZarrFactory(zarr.LMDBStore, 'lmdb_store', True),
    # ZarrFactory(zarr.DirectoryStore, 'directory_store', False),
    # ZarrFactory(zarr.LMDBStore, 'lmdb_store', False),
    # TFFactory(zarr.DirectoryStore, 'directory_store'),
    # ZarrSequenceFactory(zarr.DirectoryStore, 'directory_store'),
    # TFMultiMapFactory(zarr.DirectoryStore, 'directory_store'),
    HangarFactory(False),
    HangarFactory(True),
    TensorGeneratorFactory('gzip'),
]


def benchmark_generator(num_steps: int, gen: Generator) -> List[float]:
    times = []
    for i in range(num_steps):
        start = time.time()
        next(gen)
        times.append(time.time() - start)
        print(f'Benchmarking {(i + 1) / num_steps:.1%} done', end='\r')
        sys.stdout.flush()
    print()
    return times


def benchmark_generator_factory(
        generator_factory: GeneratorFactory,
        batch_sizes: List[int], workers: List[int],
        num_steps: int,
) -> pd.DataFrame:
    result_dfs = []
    for batch_size, num_workers in product(batch_sizes, workers):
        with generator_factory(batch_size, num_workers) as gen:
            start = time.time()
            print(f'Beginning test at batch size {batch_size}, workers {num_workers}')
            deltas = benchmark_generator(num_steps // batch_size, gen)
            print(f'Test at batch size {batch_size}, workers {num_workers} took {time.time() - start:.1f}s')
        result_df = pd.DataFrame({DELTA_COL: deltas})
        result_df[BATCH_SIZE_COL] = batch_size
        result_df[WORKER_COL] = num_workers
        result_dfs.append(result_df)
    return pd.concat(result_dfs)


class Benchmark:

    def __init__(
            self, data_descriptions: List[DataDescription], num_samples: int,
            batch_sizes: List[int], num_workers: List[int],
    ):
        self.data_descriptions = data_descriptions
        self.num_samples = num_samples
        self.batch_sizes = batch_sizes
        self.num_workers = num_workers

    def run(self, factories: List[GeneratorFactory]) -> pd.DataFrame:
        performance_dfs = []
        for factory in factories:
            os.makedirs(SYNTHETIC_DATA_PATH, exist_ok=True)
            try:
                name = factory.get_name()
                print(f'------------ {name} ------------')
                factory.setup(self.num_samples, self.data_descriptions)
                performance_df = benchmark_generator_factory(
                    factory, self.batch_sizes, self.num_workers, self.num_samples,
                )
                performance_df[NAME_COL] = name
                performance_dfs.append(performance_df)
            finally:
                print('Emptying synthetic data')
                shutil.rmtree(SYNTHETIC_DATA_PATH)
        return pd.concat(performance_dfs)


ECG_BENCHMARK = Benchmark(
    [
        ('ecg', (5000, 12), StorageType.CONTINUOUS),
        ('bmi', (1,), StorageType.CONTINUOUS),
    ],
    num_samples=4096, batch_sizes=[64, 128, 256], num_workers=[1, 2, 4, 8],
)
MRI_3D_BENCHMARK = Benchmark(
    [
        ('mri', (256, 160, 16), StorageType.CONTINUOUS),
        ('segmentation', (256, 160, 333), StorageType.CONTINUOUS),
    ],
    num_samples=256, batch_sizes=[4, 8, 16], num_workers=[1, 2, 4],
)
MRI_4D_BENCHMARK = Benchmark(
    [
        ('mri', (256, 160, 16, 1), StorageType.CONTINUOUS),
        ('segmentation', (256, 160, 16, 13), StorageType.CONTINUOUS),
    ],
    num_samples=256, batch_sizes=[4, 8, 16], num_workers=[1, 2, 4, 8],
)
ECG_MULTITASK_BENCHMARK = Benchmark(
    [('ecg', (5000, 12), StorageType.CONTINUOUS)] + [(f'interval_{i}', (1,), StorageType.CONTINUOUS) for i in range(20)],
    num_samples=4096, batch_sizes=[64, 128, 256], num_workers=[1, 2, 4, 8],
)
TEST_BENCHMARK = Benchmark(
    (
        [('ecg', (5000, 12), StorageType.CONTINUOUS)]
    ),
    num_samples=16, batch_sizes=[1, 2], num_workers=[1, 2, 4],
)
BENCHMARKS = {
    'mri_3d': MRI_3D_BENCHMARK,
    'ecg_multi_task': ECG_MULTITASK_BENCHMARK,
    'ecg_single_task': ECG_BENCHMARK,
}


def plot_benchmark(performance_df: pd.DataFrame, save_path: str, benchmark_name: str):
    plt.figure(figsize=(performance_df[BATCH_SIZE_COL].nunique() * 6, 6))
    sns.catplot(
        data=performance_df, kind='bar',
        hue=NAME_COL, y=DELTA_COL, x=WORKER_COL, col=BATCH_SIZE_COL,
    )
    plt.suptitle(f'Benchmark: {benchmark_name}', weight='bold', size='large', y=1.05)
    print(f'Saving figure to {save_path}')
    plt.savefig(save_path, dpi=200, bbox_inches='tight')


def run_benchmark(benchmark_name: str):
    performance_df = BENCHMARKS[benchmark_name].run(FACTORIES)
    output_folder = os.path.join(os.path.dirname(__file__), 'benchmark_results', benchmark_name)
    date = datetime.datetime.now().strftime('%d-%m-%Y_%H:%M:%S')
    description = f'{date}_cpus-{cpu_count()}'
    os.makedirs(output_folder, exist_ok=True)
    tsv_path = os.path.join(output_folder, f'{description}_results.tsv')
    print(f'Saving benchmark tsv to {tsv_path}')
    performance_df.to_csv(
        tsv_path, sep='\t', index=False, float_format='%.5f',
    )
    plot_benchmark(
        performance_df,
        os.path.join(output_folder, f'{description}_plot.png'),
        benchmark_name,
    )


if __name__ == '__main__':
    # TODO: add memory and line profiling
    parser = ArgumentParser()
    parser.add_argument(
        '--benchmarks', required=False, nargs='*',
        help='Benchmarks to run. If no argument is provided, all will be run.',
    )
    args = parser.parse_args()
    benchmarks = args.benchmarks or list(BENCHMARKS)
    print(f'Will run benchmarks: {", ".join(benchmarks)}')
    os.makedirs(SYNTHETIC_DATA_PATH, exist_ok=True)
    for benchmark in benchmarks:
        print('======================================')
        print(f'Running benchmark {benchmark}')
        print('======================================')
        run_benchmark(benchmark)
