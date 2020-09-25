import time
from argparse import ArgumentParser
import os
from typing import List, Generator, Callable, ContextManager
from itertools import product
import pandas as pd
from ml4h.defines import StorageType
from contextlib import contextmanager
import seaborn as sns
import matplotlib.pyplot as plt


from benchmarks.data import build_tensor_maps, build_hd5s_ukbb, get_hd5_paths


# batch_size, num_workers -> generator
GeneratorFactory = Callable[[int, int], ContextManager[Generator]]


def benchmark_generator(num_steps: int, gen: Generator) -> List[float]:
    # TODO memory profile
    times = []
    for i in range(num_steps):
        start = time.time()
        next(gen)
        times.append(time.time() - start)
        print(f'{(i + 1) / num_steps:.1%} done', end='\r')
    print()
    return times


DELTA_COL = 'step delta'
WORKER_COL = 'num workers'
BATCH_SIZE_COL = 'batch size'
NAME_COL = 'name'


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


ECG_DESCRIPTIONS = [
    ('ecg', (5000, 12), StorageType.CONTINUOUS),
    ('bmi', (1,), StorageType.CONTINUOUS),
]
NUM_ECGS = 4096


@contextmanager
def ecg_tensor_generator_factory(batch_size: int, num_workers: int):
    from ml4h.tensor_generators import TensorGenerator
    tmaps = build_tensor_maps(ECG_DESCRIPTIONS)
    gen = TensorGenerator(
        batch_size=batch_size, num_workers=num_workers,
        input_maps=tmaps, output_maps=[],
        cache_size=0, paths=get_hd5_paths(True, NUM_ECGS),
    )
    yield gen
    gen.kill_workers()
    del gen


def benchmark_ecg() -> pd.DataFrame:
    factories = [
        ('TensorGenerator_lzf_hd5', ecg_tensor_generator_factory, lambda: build_hd5s_ukbb(ECG_DESCRIPTIONS, NUM_ECGS, 'lzf')),
        ('TensorGenerator_gzip_hd5', ecg_tensor_generator_factory, lambda: build_hd5s_ukbb(ECG_DESCRIPTIONS, NUM_ECGS, 'gzip')),
    ]
    batch_sizes = [64, 128, 256]
    num_workers = [1, 2, 4, 8]
    performance_dfs = []
    for name, factory, setup in factories:
        print(f'------------ {name} ------------')
        setup()
        performance_df = benchmark_generator_factory(
            factory, batch_sizes, num_workers, NUM_ECGS
        )
        performance_df[NAME_COL] = name
        performance_dfs.append(performance_df)
    return pd.concat(performance_dfs)


def plot_benchmark(performance_df: pd.DataFrame, save_path: str):
    performance_df['samples / sec'] = 1 / performance_df[DELTA_COL] * performance_df[BATCH_SIZE_COL]
    plt.figure(figsize=(performance_df[BATCH_SIZE_COL].nunique() * 6, 6))
    sns.catplot(
        data=performance_df, kind='point',
        hue=NAME_COL, y='samples / sec', x=WORKER_COL, col=BATCH_SIZE_COL,
    )
    plt.savefig(save_path, dpi=200)


def run_benchmark(benchmark_name: str, output_folder: str):
    # TODO: include got commit, datetime in output name, number of cpus
    os.makedirs(output_folder, exist_ok=True)
    benchmarks = {
        'ecg': benchmark_ecg,
    }
    performance_df = benchmarks[benchmark_name]()
    performance_df.to_csv(
        os.path.join(output_folder, f'{benchmark_name}.tsv'),
        sep='\t', index=False,
    )
    plot_benchmark(performance_df, os.path.join(output_folder, f'{benchmark_name}.png'))


if __name__ == '__main__':
    # TODO: add memory and line profiling
    parser = ArgumentParser()
    parser.add_argument('--benchmark', help='Benchmark to run.')
    parser.add_argument('--output_folder', help='Where to save the benchmark results.')
    args = parser.parse_args()
    run_benchmark(args.benchmark, args.output_folder)
