import io
import itertools
import os
import time
from collections import defaultdict
from datetime import datetime
import av
import lmdb
import numpy as np
import pandas as pd

# =====================================================================
# CONFIGURATION VARIABLES (Set your test parameters here)
# =====================================================================
WIDE_FILE = "/mnt/disks/droid-af/data/droid_af_wide_trainvalid_2026_07_16.pq"
LOCAL_LMDB_DIR = "/mnt/disks/mgb-echo/stage3_trainvalid/"  # Path to directory containing {study}.lmdb

# Column name in the Parquet file containing the sample IDs
SAMPLE_ID_COL = "sample_id"

# Number of times to repeat the benchmark
NUM_RUNS = 50

START_FRAME = 0
RANDOMIZE_START_FRAME = True
NFRAMES = 16
SKIP_MODULO = 1
LOADING_OPTION = None


# Add any custom transforms here if needed
def dummy_transform(video, loading_option=None):
    return video


TRANSFORMS = [dummy_transform]
# =====================================================================


def log_step(step_name, start_time=None, step_metrics=None):
    """Helper to log timestamps before/after operations and accumulate timings."""
    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    if start_time is not None:
        elapsed_ms = (time.perf_counter() - start_time) * 1000
        print(f"[{timestamp}] FINISHED: {step_name} ({elapsed_ms:.2f} ms)")
        if step_metrics is not None:
            step_metrics[step_name].append(elapsed_ms)
        return elapsed_ms
    else:
        print(f"[{timestamp}] STARTING: {step_name}")
        return time.perf_counter()


class DataFetcherTester:

    def __init__(self):
        self.local_lmdb_dir = LOCAL_LMDB_DIR
        self.start_frame = START_FRAME
        self.randomize_start_frame = RANDOMIZE_START_FRAME
        self.nframes = NFRAMES
        self.skip_modulo = SKIP_MODULO
        self.transforms = TRANSFORMS

    def get_raw_data(self, sample_id, loading_option=None, step_metrics=None):
        try:
            sample_id = sample_id.decode("UTF-8")
        except (UnicodeDecodeError, AttributeError):
            pass
        _, study, view = sample_id.split("_")

        lmdb_folder = os.path.join(self.local_lmdb_dir, f"{study}.lmdb")

        # --- CRITICAL OPERATION 1: Open LMDB Environment ---
        t = log_step("1. Open LMDB Environment", step_metrics=step_metrics)
        env = lmdb.open(lmdb_folder, readonly=True, lock=False)
        log_step("1. Open LMDB Environment", t, step_metrics=step_metrics)

        frames = []

        # --- CRITICAL OPERATION 2: Start Transaction & Fetch Bytes ---
        t = log_step(
            "2. LMDB Transaction & Memory Fetch", step_metrics=step_metrics
        )
        with env.begin(buffers=True) as txn:
            in_mem_bytes_io = io.BytesIO(txn.get(view.encode("utf-8")))
            log_step(
                "2. LMDB Transaction & Memory Fetch",
                t,
                step_metrics=step_metrics,
            )

            # --- CRITICAL OPERATION 3: Open PyAV Video Container ---
            t = log_step(
                "3. Open PyAV Video Container", step_metrics=step_metrics
            )
            video_container = av.open(
                in_mem_bytes_io, metadata_errors="ignore"
            )
            stream = video_container.streams.video[0]
            stream.thread_type = "AUTO"
            log_step(
                "3. Open PyAV Video Container", t, step_metrics=step_metrics
            )

            start_frame = self.start_frame

            # --- CRITICAL OPERATION 4: Frame Count Calculation & Randomization ---
            if self.randomize_start_frame:
                t = log_step(
                    "4. Calculate Frame Count & Randomize Start Frame",
                    step_metrics=step_metrics,
                )
                total_frames = stream.frames
                if not total_frames:
                    total_frames = sum(
                        1 for _ in video_container.decode(video=0)
                    )
                    video_container.seek(0)
                frame_range = total_frames - (self.nframes * self.skip_modulo)
                if frame_range > 0:
                    start_frame = np.random.randint(frame_range)
                log_step(
                    "4. Calculate Frame Count & Randomize Start Frame",
                    t,
                    step_metrics=step_metrics,
                )

            # --- CRITICAL OPERATION 5: Frame Decoding & Array Extraction ---
            t = log_step("5. Decode Video Frames", step_metrics=step_metrics)
            video_frames = itertools.cycle(video_container.decode(video=0))
            for i, frame in enumerate(video_frames):
                if len(frames) == self.nframes:
                    break
                if i < start_frame:
                    continue
                if self.skip_modulo > 1:
                    if ((i - start_frame) % self.skip_modulo) != 0:
                        continue
                frame = frame.to_ndarray(format="rgb24")
                frames.append(frame)
            del video_frames
            video_container.close()
            log_step("5. Decode Video Frames", t, step_metrics=step_metrics)

        env.close()

        # --- CRITICAL OPERATION 6: NumPy Array Normalization ---
        t = log_step(
            "6. Convert & Normalize NumPy Array", step_metrics=step_metrics
        )
        video = np.array(frames, dtype="float32") / 255.0
        log_step(
            "6. Convert & Normalize NumPy Array", t, step_metrics=step_metrics
        )

        # --- CRITICAL OPERATION 7: Apply Transforms ---
        t = log_step("7. Apply Transforms", step_metrics=step_metrics)
        for transform in self.transforms:
            video = transform(video, loading_option)
        log_step("7. Apply Transforms", t, step_metrics=step_metrics)

        return np.squeeze(np.asarray(video, dtype="float32"))


if __name__ == "__main__":
    print("==========================================")
    print(f"  STARTING GET_RAW_DATA TEST ({NUM_RUNS} RUNS) ")
    print("==========================================\n")

    # --- Load Parquet & Get Sample IDs ---
    print(f"Loading dataset from: {WIDE_FILE}")
    df = pd.read_parquet(WIDE_FILE)

    if SAMPLE_ID_COL in df.columns:
        sample_ids = df[SAMPLE_ID_COL].dropna().values
    else:
        sample_ids = df.iloc[:, 0].dropna().values

    tester = DataFetcherTester()
    step_metrics = defaultdict(list)
    total_times = []
    successful_runs = 0

    # --- Run Loop ---
    for run_idx in range(1, NUM_RUNS + 1):
        selected_sample_id = str(np.random.choice(sample_ids))
        print(
            f"\n--- RUN {run_idx}/{NUM_RUNS} | Sample ID: {selected_sample_id} ---"
        )

        total_start = time.perf_counter()
        try:
            output = tester.get_raw_data(
                selected_sample_id, LOADING_OPTION, step_metrics=step_metrics
            )
            total_elapsed = (time.perf_counter() - total_start) * 1000
            total_times.append(total_elapsed)
            successful_runs += 1
            print(
                f"Run {run_idx} finished in {total_elapsed:.2f} ms | Output Shape: {output.shape}"
            )

        except Exception as e:
            print(f"[ERROR] Run {run_idx} failed with exception: {e}")

    # --- Final Summary Report ---
    print("\n" + "=" * 66)
    print(
        f"           BENCHMARK SUMMARY ({successful_runs}/{NUM_RUNS} SUCCESSFUL RUNS)"
    )
    print("=" * 66)

    if successful_runs > 0:
        print(f"{'Step Name':<50} | {'Avg Time (ms)':<12}")
        print("-" * 66)
        for step_name, times in step_metrics.items():
            avg_time = np.mean(times)
            print(f"{step_name:<50} | {avg_time:>10.2f} ms")
        print("-" * 66)
        print(
            f"{'TOTAL EXECUTION TIME (AVG)':<50} | {np.mean(total_times):>10.2f} ms"
        )
    else:
        print("No successful runs to compute statistics.")
    print("=" * 66)