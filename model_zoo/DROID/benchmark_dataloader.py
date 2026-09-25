"""Benchmark the actual finite DROID dataset against the loader at 06dc8a3.

Every case runs in a fresh process. Timings exclude label preparation and one
warmup epoch. These are warm-cache loader measurements, not GPU training speeds.
"""

import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import threading
import time

import numpy as np
import pandas as pd
import psutil


def reference_dataset(loader, output, ids, batch_size, signature):
    """Original serial batch generator, including per-epoch pandas lookups."""
    import tensorflow as tf

    def generator():
        for start in range(0, len(ids) - batch_size + 1, batch_size):
            images, labels = [], []
            for sample_id in ids[start:start + batch_size]:
                images.append(loader.get_raw_data(sample_id))
                labels.append(output.get_raw_data(sample_id))
            yield np.stack(images).astype(np.float32, copy=False), np.stack(labels).astype(np.float32, copy=False)
    return tf.data.Dataset.from_generator(generator, output_signature=signature)


class PeakMemory:
    def __enter__(self):
        self.process = psutil.Process()
        self.peak = self.process.memory_info().rss
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.sample, daemon=True)
        self.thread.start()
        return self

    def sample(self):
        while not self.stop.wait(0.02):
            self.peak = max(self.peak, self.process.memory_info().rss)

    def __exit__(self, *args):
        self.stop.set()
        self.thread.join()


def run_case(args):
    import tensorflow as tf
    from benchmark_reference import ReferenceVideoDescription
    from data_descriptions.echo import LmdbEchoStudyVideoDataDescription
    from data_descriptions.echo_dataset import make_dataset
    from data_descriptions.video_io import read_video_bytes
    from data_descriptions.wide_file import EcholabDataDescription

    tf.config.threading.set_inter_op_parallelism_threads(2)
    tf.config.threading.set_intra_op_parallelism_threads(1)
    # Explicit wide-file input is needed for benchmarking the user's real LMDBs.
    df = pd.read_parquet(args.wide_file, columns=['sample_id']).drop_duplicates()
    if args.samples:
        df = df.sample(n=min(args.samples, len(df)), random_state=args.seed)
    ids = df['sample_id'].astype(str).tolist()
    np.random.default_rng(args.seed).shuffle(ids)
    df['benchmark_label_1'] = np.arange(len(df), dtype=np.float32)
    df['benchmark_label_2'] = 1.0
    output = EcholabDataDescription(df, 'sample_id',
                                   ['benchmark_label_1', 'benchmark_label_2'], 'labels')
    kwargs = dict(nframes=args.nframes, skip_modulo=args.skip_modulo,
                  randomize_start_frame=args.randomize_start_frame)
    if args.case == 'legacy':
        loader = ReferenceVideoDescription(args.lmdb_dir, 'reference', **kwargs)
    else:
        mode = 'sequential' if args.case.startswith('parallel_sequential') else 'auto'
        loader = LmdbEchoStudyVideoDataDescription(args.lmdb_dir, 'optimized',
                                                  decode_mode=mode, decode_threads=1, **kwargs)
    signature = (tf.TensorSpec((args.batch_size, args.nframes, 224, 224, 3), tf.float32),
                 tf.TensorSpec((args.batch_size, 2), tf.float32))
    started = time.perf_counter()
    if args.case in ('legacy', 'serial_auto'):
        dataset = reference_dataset(loader, output, ids, args.batch_size, signature)
    else:
        workers = int(args.case.rsplit('_w', 1)[1])
        dataset = make_dataset(loader, output, ids, args.batch_size, signature, False, workers)
    dataset = dataset.prefetch(args.prefetch_batches)
    setup = time.perf_counter() - started
    first_pass = time.perf_counter()
    for batch in dataset:
        del batch
    first_pass_seconds = time.perf_counter() - first_pass
    # Estimate warm storage read/copy overhead separately from decode/tf.data.
    begin = time.perf_counter()
    byte_count = 0
    for sample_id in ids:
        byte_count += len(read_video_bytes(args.lmdb_dir, sample_id))
    read_seconds = time.perf_counter() - begin
    gc.collect()
    process = psutil.Process()
    fds_start = process.num_fds()
    rss_start = process.memory_info().rss
    epochs, waits, rss = [], [], []
    np.random.seed(args.seed)
    if args.profile_dir:
        tf.profiler.experimental.start(str(Path(args.profile_dir) / args.case))
    with PeakMemory() as memory:
        for _ in range(args.repeats):
            start = time.perf_counter()
            iterator = iter(dataset)
            samples = 0
            while True:
                before = time.perf_counter()
                try:
                    images, labels = next(iterator)
                except StopIteration:
                    break
                waits.append(time.perf_counter() - before)
                samples += int(images.shape[0])
                del images, labels
            epochs.append(dict(seconds=time.perf_counter() - start, samples=samples))
            del iterator
            gc.collect()
            rss.append(process.memory_info().rss / 2**20)
    if args.profile_dir:
        tf.profiler.experimental.stop()
    if not samples:
        raise ValueError('Need enough samples for at least one batch')
    rates = [epoch['samples'] / epoch['seconds'] for epoch in epochs]
    return dict(
        case=args.case, samples_per_second=statistics.median(rates),
        seconds_per_batch=args.batch_size / statistics.median(rates),
        rates=rates, epochs=epochs, next_batch_p50_ms=float(np.percentile(waits, 50) * 1000),
        next_batch_p95_ms=float(np.percentile(waits, 95) * 1000), setup_seconds=setup,
        first_pass_seconds=first_pass_seconds,
        peak_rss_mib=memory.peak / 2**20, rss_after_warmup_mib=rss_start / 2**20,
        rss_after_epochs_mib=rss, fd_delta=process.num_fds() - fds_start,
        warm_read_mib_per_second=byte_count / 2**20 / read_seconds,
        warm_read_ms_per_clip=1000 * read_seconds / len(ids),
        mean_encoded_mib=byte_count / 2**20 / len(ids),
        tensorflow_devices=[device.name for device in tf.config.list_physical_devices()],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lmdb-dir', required=True)
    parser.add_argument('--wide-file', required=True, help='Parquet with sample_id; no clinical labels read.')
    parser.add_argument('--output', required=True, help='New JSON result file.')
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--nframes', type=int, default=32)
    parser.add_argument('--skip-modulo', type=int, default=4)
    parser.add_argument('--samples', type=int, default=96)
    parser.add_argument('--workers', type=int, nargs='+', default=[1, 2, 4, 8])
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--prefetch-batches', type=int, default=1)
    parser.add_argument('--randomize-start-frame', action='store_true')
    parser.add_argument('--profile-dir', help='Optional TensorFlow trace directory; affects timings.')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--case', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if (min(args.batch_size, args.nframes, args.skip_modulo, args.repeats, *args.workers) < 1
            or args.samples < 0 or args.prefetch_batches < 0):
        parser.error('Invalid dimensions, repeats, workers or prefetch count')
    if args.case:
        print(json.dumps(run_case(args)))
        return
    if Path(args.output).exists():
        parser.error('Output already exists; choose a new result path.')
    cases = ['legacy', 'serial_auto', 'parallel_sequential_w4']
    cases += [f'parallel_auto_w{worker}' for worker in args.workers]
    results = []
    for case in cases:
        command = [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:], '--case', case]
        environment = dict(os.environ, TF_CPP_MIN_LOG_LEVEL='2')
        completed = subprocess.run(command, env=environment, capture_output=True, text=True)
        if completed.returncode:
            sys.stderr.write(completed.stderr)
            raise RuntimeError(f'{case} failed')
        result = json.loads(completed.stdout.strip().splitlines()[-1])
        results.append(result)
        print(f"{case:26} {result['samples_per_second']:7.2f} clips/s  "
              f"{result['seconds_per_batch']:.4f} s/batch  "
              f"peak RSS {result['peak_rss_mib']:.0f} MiB", flush=True)
    report = dict(
        configuration=vars(args), platform=platform.platform(), cpu_count=os.cpu_count(),
        cpu_model=platform.processor(), system_memory_gib=psutil.virtual_memory().total / 2**30,
        versions={name: importlib.metadata.version(name)
                  for name in ['numpy', 'av', 'lmdb', 'tensorflow', 'pandas', 'ml4ht']},
        tf_use_legacy_keras=os.environ.get('TF_USE_LEGACY_KERAS', '0'),
        source_sha256={name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
                       for name in ['benchmark_dataloader.py', 'benchmark_reference.py',
                                    'data_descriptions/echo.py', 'data_descriptions/echo_dataset.py',
                                    'data_descriptions/video_io.py']},
        cache_condition='One full warmup epoch per case; warm OS cache; local synthetic data unless specified.',
        results=results,
    )
    with open(args.output, 'x') as output:
        json.dump(report, output, indent=2)
    best = max(results, key=lambda result: result['samples_per_second'])
    print(f"Best: {best['case']}, {best['samples_per_second'] / results[0]['samples_per_second']:.2f}x legacy")
    print(f'Results: {args.output}')


if __name__ == '__main__':
    main()
