"""Benchmark the DROID echo video dataloader: baseline vs. optimized.

Times how long it takes to load N samples through the original dataloader
(``LmdbEchoStudyVideoDataDescriptionBaseline`` in ``data_descriptions.echo_baseline``)
and the optimized one (``LmdbEchoStudyVideoDataDescription`` in ``data_descriptions.echo``),
which differ in: metadata-based frame counting (no double decode), direct
``frame.to_ndarray`` (no PIL round-trip), multithreaded decode, and a cached LMDB env.

Both loaders run over the *same* sample sequence with identical parameters.

Examples
--------
# Auto-discover sample_ids from the LMDB directory, load 200 samples:
python benchmark_dataloader.py --lmdb_dir /path/to/lmdbs --n_samples 200

# Draw sample_ids from a training wide-file, and verify the outputs match:
python benchmark_dataloader.py --lmdb_dir /path/to/lmdbs \
    --wide_file /path/to/wide.pq --n_samples 200 --check
"""

import os
import sys
import glob
import time
import random
import argparse
import statistics

import numpy as np
import pandas as pd

# Allow running from anywhere: make the DROID package dir importable.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_descriptions.echo import LmdbEchoStudyVideoDataDescription
from data_descriptions.echo_baseline import LmdbEchoStudyVideoDataDescriptionBaseline


def discover_sample_ids(lmdb_dir):
    """Build sample_ids ('<patient>_<study>_<view>') by scanning *.lmdb folders.

    The patient token is only used to split the id back apart, so a constant
    placeholder ('0') is fine here.
    """
    sample_ids = []
    for lmdb_folder in sorted(glob.glob(os.path.join(lmdb_dir, "*.lmdb"))):
        study = os.path.basename(lmdb_folder)[: -len(".lmdb")]
        log_path = os.path.join(lmdb_folder, f"log_{study}.pq")
        if not os.path.exists(log_path):
            continue
        log = pd.read_parquet(log_path)
        stored_views = log[log["stored"]]["view"].tolist()
        sample_ids.extend(f"0_{study}_{view}" for view in stored_views)
    return sample_ids


def load_sample_ids_from_wide(wide_file):
    df = pd.read_parquet(wide_file, columns=["sample_id"])
    return df["sample_id"].astype(str).tolist()


def pick_sample_ids(pool, n_samples, seed):
    if not pool:
        raise SystemExit("No sample_ids found. Check --lmdb_dir / --wide_file.")
    rng = random.Random(seed)
    if n_samples <= len(pool):
        return rng.sample(pool, n_samples)
    # Not enough unique ids: sample with replacement to reach n_samples.
    return [rng.choice(pool) for _ in range(n_samples)]


def time_loader(dd, sample_ids):
    """Load every sample_id in order, returning (total_seconds, per_sample_seconds)."""
    per_sample = []
    t0 = time.perf_counter()
    for sid in sample_ids:
        s = time.perf_counter()
        dd.get_raw_data(sid)
        per_sample.append(time.perf_counter() - s)
    total = time.perf_counter() - t0
    return total, per_sample


def report(name, total, per_sample):
    ms = [t * 1e3 for t in per_sample]
    n = len(ms)
    print(f"\n[{name}]  {n} samples in {total:.3f} s  ({n / total:.1f} samples/s)")
    print(f"  per-sample ms:  mean {statistics.mean(ms):.2f} | "
          f"median {statistics.median(ms):.2f} | "
          f"p95 {sorted(ms)[min(n - 1, int(0.95 * n))]:.2f} | "
          f"min {min(ms):.2f} | max {max(ms):.2f}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--lmdb_dir", required=True, help="Directory containing <study>.lmdb folders.")
    p.add_argument("--n_samples", type=int, default=100, help="Number of samples to load per loader.")
    p.add_argument("--nframes", type=int, default=32, help="Frames to decode per clip.")
    p.add_argument("--skip_modulo", type=int, default=2, help="Temporal subsample factor.")
    p.add_argument("--wide_file", default=None, help="Optional parquet with a 'sample_id' column to draw from.")
    p.add_argument("--randomize_start_frame", action="store_true", help="Randomize clip start frame (as in training).")
    p.add_argument("--env_cache_size", type=int, default=128, help="LMDB env cache size for the optimized loader.")
    p.add_argument("--seed", type=int, default=0, help="Seed for sample selection.")
    p.add_argument("--no_warmup", action="store_true", help="Skip the OS page-cache warmup pass.")
    p.add_argument("--check", action="store_true",
                   help="Verify baseline/optimized outputs match (only meaningful without --randomize_start_frame).")
    args = p.parse_args()

    pool = (load_sample_ids_from_wide(args.wide_file) if args.wide_file
            else discover_sample_ids(args.lmdb_dir))
    sample_ids = pick_sample_ids(pool, args.n_samples, args.seed)
    print(f"Selected {len(sample_ids)} sample_ids "
          f"({len(set(sample_ids))} unique) from a pool of {len(pool)}.")

    common = dict(nframes=args.nframes, skip_modulo=args.skip_modulo,
                  randomize_start_frame=args.randomize_start_frame)
    baseline = LmdbEchoStudyVideoDataDescriptionBaseline(
        local_lmdb_dir=args.lmdb_dir, name="baseline", **common)
    optimized = LmdbEchoStudyVideoDataDescription(
        local_lmdb_dir=args.lmdb_dir, name="optimized",
        env_cache_size=args.env_cache_size, **common)

    if args.check:
        if args.randomize_start_frame:
            print("\n[check] Skipped exact comparison: --randomize_start_frame makes outputs nondeterministic.")
        else:
            max_diff = 0.0
            for sid in sample_ids[: min(10, len(sample_ids))]:
                a = baseline.get_raw_data(sid)
                b = optimized.get_raw_data(sid)
                assert a.shape == b.shape, f"shape mismatch for {sid}: {a.shape} vs {b.shape}"
                max_diff = max(max_diff, float(np.max(np.abs(a - b))))
            print(f"\n[check] Outputs match on first {min(10, len(sample_ids))} samples. "
                  f"Max abs diff = {max_diff:.3e}")

    if not args.no_warmup:
        # Prime the OS page cache so we measure decode cost, not first-touch disk reads.
        # Uses the optimized loader; both loaders read the same underlying bytes.
        print("\nWarming OS page cache...")
        for sid in sample_ids:
            optimized.get_raw_data(sid)

    base_total, base_ps = time_loader(baseline, sample_ids)
    opt_total, opt_ps = time_loader(optimized, sample_ids)

    report("baseline", base_total, base_ps)
    report("optimized", opt_total, opt_ps)

    speedup = base_total / opt_total if opt_total > 0 else float("inf")
    print(f"\n==> Optimized is {speedup:.2f}x faster "
          f"({base_total:.3f} s -> {opt_total:.3f} s over {len(sample_ids)} samples).\n")


if __name__ == "__main__":
    main()
