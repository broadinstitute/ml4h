# DROID video loading

The training recipe now decodes clips concurrently and skips unselected MJPEG
frames. Existing per-study LMDB files work unchanged. For a V100 with 8–12 CPUs
and batch size 4, start with the new defaults:

```bash
# Append these to your existing training command (they are also the defaults):
--loader_workers 4 --video_decode_mode auto --video_decode_threads 1 --prefetch_batches 1
```

These settings do not increase the model batch size or change the backbone. They
increase host-side parallelism and can increase host RAM usage. `--loader_workers
2` reduces the parallel buffers; `--prefetch_batches 0` disables explicit prefetch.
`--video_decode_mode sequential` is available for decoder troubleshooting.
One float32 batch at 4×32×224×224×3 is 73.5 MiB; worker outputs, decoder buffers,
TensorFlow, the model, and prefetched batches add to that. Precomputed labels
also scale with the number of examples and output dimensions.

## What was slow

At starting commit `06dc8a3`, the recipe's Python generator read and decoded every
video in a batch serially, performed a pandas label lookup for each example, and
stacked complete float32 clips into a new batch. Prefetching one batch overlapped
this serial producer with training but did not parallelize it.

The decoder used `itertools.cycle(container.decode(...))`: it decoded skipped
frames, retained them in the cycle, and converted only selected frames to RGB.
For a 32-frame clip at stride 4, roughly 125 frames plus the random starting
offset were decoded. FFmpeg also chose its own decoder thread count.

The new path:

1. Builds label tensors once, then shuffles sample IDs and their labels together.
2. Uses a finite `tf.data` parallel map to read/decode clips, then batches them.
3. For indexed MJPEG AVI only, checks that every frame has an independent packet
   and decodes only the requested frame indices. Short clips retain cyclic
   padding, and repeated frame indices are decoded once.
4. Uses generic sequential decoding for other formats or ambiguous MJPEG indexes.
5. Runs existing per-clip augmentations on the CPU, keeping shared temporal
   augmentation parameters and avoiding eager work on the training GPU.

There is no persistent LMDB environment cache. Environments and transactions are
closed before decoding, including on errors. Fixed lock stripes protect reads of
the same study against duplicate concurrent environment opens. Decoder workers
and prefetch are explicitly bounded; validation stays finite and training repeats
only after batching, preserving dropped remainder and epoch reshuffling.
Parallel scheduling changes which sample consumes each global RNG draw; a fixed
NumPy seed does not reproduce the former sample-by-sample random crop sequence.
Validation remains deterministic. The crop distribution and exclusive random
start upper bound are unchanged.

References: [TensorFlow input performance](https://www.tensorflow.org/guide/data_performance),
[PyAV packets](https://pyav.org/docs/stable/api/packet.html),
[LMDB environment and resource rules](https://lmdb.readthedocs.io/en/release/).

## Measured local results (2026-09-18)

ARM64 Mac, 14 logical CPUs, 36 GiB RAM; TensorFlow 2.19 / legacy Keras, PyAV 18.1,
LMDB 2.3, NumPy 2.1.3. 96 distinct 160-frame synthetic videos (42 MiB on disk),
batch size 4, 32 selected frames, random start, one prefetched batch, no
augmentations. Medians of three warm-cache passes in isolated processes:

| Pipeline | Stride 4 clips/s | Stride 2 clips/s |
| --- | ---: | ---: |
| Original serial generator + original decoder | 41.11 | 44.51 |
| Serial generator + selected-frame decoder | 59.89 | 58.10 |
| Parallel map, 4 workers, generic sequential decoder | 50.12 | 91.32 |
| Selected-frame decoder, 1 worker | 58.94 | 59.76 |
| Selected-frame decoder, 2 workers | 90.19 | 89.61 |
| Selected-frame decoder, 4 workers | **144.13** | **145.86** |
| Selected-frame decoder, 8 workers | 131.83 | 135.05 |

The four-worker configuration delivered **3.51× / 3.28×** the original throughput:
about **28 / 27 ms per four-clip batch**, compared with **97 / 90 ms**. Eight
workers were slower and used more memory. Sampled peak process RSS at stride 4
was 704 MiB for the original loader, 815 MiB for four workers, and 956 MiB for
eight. All cases had zero net descriptor growth over the measured passes. These
are total process RSS figures, including TensorFlow, not decoder-only memory.

Warm LMDB read/copy was a small fraction of a millisecond per clip locally;
decoding and scheduling dominated this synthetic benchmark. This does **not**
establish the bottleneck on GCP or predict a 3.5× improvement in end-to-end
training. The original loader was already much faster locally than the reported
3-second training step, so the actual job needs the profiler check below.

Raw results with runtime versions, source hashes, and all per-pass measurements:
[stride 4](benchmark_results/20260918_32frames_stride4.json),
[stride 2](benchmark_results/20260918_32frames_stride2.json).

A separate [20-pass run](benchmark_results/20260918_sustained_20passes.json)
loaded 1,920 clips at a median 144 clips/s. RSS rose from 808 to about 832 MiB
as buffers warmed up, then stabilized; descriptor count did not increase. This
checks repeated iterator cleanup over a small dataset, not long-term production
memory use across millions of studies.

Verification: 19 tests passed with TF 2.19 / legacy Keras, covering exact frame
and pixel equivalence, production-style OpenCV MJPEG encoding, random starts,
short-clip cycling, single-frame dimensions, MPEG4 fallback, selective decode
counts, concurrent same-study reads, error cleanup, classification/survival label
alignment, reshuffling, augmentation, and three-epoch Keras training/validation.

## Reproduce with synthetic data

Run from the repository root in the DROID environment. Benchmark-only additional
packages are `psutil` and `pytest`; generation also uses the existing PyArrow,
pandas, NumPy, PyAV, and LMDB dependencies. Tests below use TF 2.19 with legacy Keras.

```bash
python model_zoo/DROID/synthetic_lmdb.py \
  --output /tmp/droid-synthetic \
  --studies 24 --views 4 --frames 160

TF_USE_LEGACY_KERAS=1 python model_zoo/DROID/benchmark_dataloader.py \
  --lmdb-dir /tmp/droid-synthetic/lmdb \
  --wide-file /tmp/droid-synthetic/wide.pq \
  --output /tmp/droid-loader-results.json \
  --batch-size 4 --nframes 32 --skip-modulo 4 \
  --randomize-start-frame --workers 1 2 4 8 --repeats 3

TF_USE_LEGACY_KERAS=1 python -m pytest -q model_zoo/DROID/test_video_loading.py
```

Generation refuses to overwrite an existing directory. It writes grayscale,
moving, textured sectors encoded as 224×224 MJPEG AVIs; no clinical data is used.
Every study is `<study>.lmdb`, each DICOM-style view key holds an entire AVI, and
`log_<study>.pq` records stored views. `wide.pq` contains matching
`<patient>_<study>_<view>` sample IDs and synthetic labels; `splits.json` provides
disjoint patient splits. The default set contains 96 distinct videos.

The benchmark compares the **current starting revision**, not the older, slower
`echo_baseline.py`. Its frozen loader is `benchmark_reference.py`. Each case runs
in a separate process, includes one full warmup epoch, then reports median
throughput across repeated finite epochs, batch wait percentiles, sampled peak
RSS, per-epoch RSS, descriptor changes, and warm LMDB read/copy timing. Reported
throughput excludes dataset construction and label preparation, which are timed
separately. First-pass time is recorded but must not be called a cold-cache test:
the OS cache is never flushed. No augmentations are enabled in these timings.

## Verify the bottleneck on the V100

Your reported 4 clips per 3-second training step consumes 1.33 clips/s. This is
only a lower-bound throughput target: training can consume data faster once
input stalls are removed. Use your actual `n_input_frames`, `skip_modulo`, and
random-start settings when benchmarking.

Run the same benchmark **inside the training container, on the same GCP host
and shared SSD**, changing `--lmdb-dir` and `--wide-file` to the existing inputs.
Only the `sample_id` column is read from the wide file; benchmark labels are
synthetic, and reports contain aggregate measurements, not sample IDs. Supply a
wide file containing the same eligible views as training. `--samples 1000` gives
a larger sample; `--samples 0` uses every unique ID. To measure storage beyond the
page cache, use a working set larger than host RAM. Do not infer shared-SSD or
cold-cache throughput from the small synthetic set.

Then use the existing training TensorBoard profiler, which already captures
batches 20–30 (`model_descriptions/echo.py`). Compare input wait/`IteratorGetNext`
time and device compute in that trace after warmup. A short loader-only batch
time is evidence of capacity, but GPU-limited training requires seeing minimal
input stalls during **actual training**, with its augmentations and CPU/GPU
contention. The local benchmarks do not establish this on a V100.

If GCP's warm read/copy time is small but decoding is slow, tune workers 2/4/8
using the measured results. If storage time dominates or only first passes are
slow, local-SSD staging may help; the new decoder still reads the complete
compressed AVI value. No format migration or GPU decoder is needed for the
implemented changes.

## Earlier experiments recovered

The local Git history retains `a08d477` ("test dataloader optimizations + benchmark
script"). `5837926` removed that benchmark and reverted LMDB environment caching
("revert lmdb env changes to fix memory leak"). The cached remote-tracking branch
`origin/decord-test`, at `7d743ba`, also retains subsequent TFRecord/DALI experiments.
No remote was fetched or pushed for this work. This implementation is based on
`maa-droid` (`3ca2e43`, "Reduce echo training prefetch buffer").
