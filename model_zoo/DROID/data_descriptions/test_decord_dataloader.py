import io
import os
import time
from collections import defaultdict
from datetime import datetime
import lmdb
import numpy as np
import pandas as pd
import tensorflow as tf

# Import NVIDIA DALI modules
import nvidia.dali.fn as fn
import nvidia.dali.types as types
from nvidia.dali.pipeline import pipeline_def

# Resolve the correct DALI video decoder function across different DALI versions
if hasattr(fn, "decoders") and hasattr(fn.decoders, "video"):
    dali_decode_video = fn.decoders.video
elif hasattr(fn, "experimental") and hasattr(fn.experimental, "decoders") and hasattr(fn.experimental.decoders, "video"):
    dali_decode_video = fn.experimental.decoders.video
elif hasattr(fn, "plugin") and hasattr(fn.plugin, "video") and hasattr(fn.plugin.video, "decoder"):
    dali_decode_video = fn.plugin.video.decoder
else:
    raise AttributeError("Could not locate DALI video decoder in installed DALI version.")

# =====================================================================
# CONFIGURATION VARIABLES (Set your test parameters here)
# =====================================================================
WIDE_FILE = "/mnt/disks/droid-af/data/droid_af_wide_trainvalid_2026_07_16.pq"
LOCAL_LMDB_DIR = "/mnt/disks/mgb-echo/stage3_trainvalid/"  # Path to directory containing {study}.lmdb

# Column name in the Parquet file containing the sample IDs
SAMPLE_ID_COL = "sample_id"

# GPU Device Index
GPU_INDEX = 0

# Number of times to repeat the benchmark
NUM_RUNS = 10

START_FRAME = 0
RANDOMIZE_START_FRAME = True
NFRAMES = 16
SKIP_MODULO = 1
LOADING_OPTION = None


# Custom GPU TensorFlow transforms
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

    def __init__(self, gpu_index=0):
        self.local_lmdb_dir = LOCAL_LMDB_DIR
        self.start_frame = START_FRAME
        self.randomize_start_frame = RANDOMIZE_START_FRAME
        self.nframes = NFRAMES
        self.skip_modulo = SKIP_MODULO
        self.transforms = TRANSFORMS
        self.gpu_index = gpu_index

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

        # --- CRITICAL OPERATION 2: Start Transaction & Fetch Bytes ---
        t = log_step("2. LMDB Transaction & Memory Fetch", step_metrics=step_metrics)
        with env.begin(buffers=True) as txn:
            raw_bytes = bytes(txn.get(view.encode("utf-8")))
            video_array = np.frombuffer(raw_bytes, dtype=np.uint8)
        env.close()
        log_step("2. LMDB Transaction & Memory Fetch", t, step_metrics=step_metrics)

        # --- CRITICAL OPERATION 3: Decode Video with NVIDIA DALI ---
        t = log_step("3. Decode Video Stream via DALI GPU Reader", step_metrics=step_metrics)
        
        # Define DALI pipeline for decoding video byte buffer
        @pipeline_def(batch_size=1, num_threads=2, device_id=self.gpu_index)
        def video_decode_pipeline():
            # batch=False tells DALI video_array is a single sample, not a batch of scalars
            encoded = fn.external_source(source=[video_array], batch=False, dtype=types.UINT8)
            decoded_video = dali_decode_video(encoded, device="mixed")
            return decoded_video

        pipe = video_decode_pipeline()
        pipe.build()
        pipe_output = pipe.run()
        
        # Extract DALI output tensor (Shape: [F, H, W, C])
        dali_tensor = pipe_output[0][0]

        # Convert DALI GPU tensor directly to TensorFlow GPU Tensor via DLPack
        try:
            raw_frames = tf.experimental.dlpack.from_dlpack(dali_tensor.as_dlpack())
        except Exception:
            raw_frames = tf.constant(dali_tensor.as_cpu().as_array())

        log_step("3. Decode Video Stream via DALI GPU Reader", t, step_metrics=step_metrics)

        # --- CRITICAL OPERATION 4: Frame Count Calculation & Randomization ---
        t = log_step("4. Calculate Frame Count & Randomize Start Frame", step_metrics=step_metrics)
        total_frames = int(raw_frames.shape[0])
        start_frame = self.start_frame

        if self.randomize_start_frame:
            frame_range = total_frames - (self.nframes * self.skip_modulo)
            if frame_range > 0:
                start_frame = np.random.randint(frame_range)
        log_step("4. Calculate Frame Count & Randomize Start Frame", t, step_metrics=step_metrics)

        # --- CRITICAL OPERATION 5: Slice Requested Frames ---
        t = log_step("5. Extract Frame Batch", step_metrics=step_metrics)
        end_frame = start_frame + (self.nframes * self.skip_modulo)
        frame_indices = list(range(start_frame, end_frame, self.skip_modulo))
        
        # Slice frames directly on GPU in TensorFlow
        frames = tf.gather(raw_frames, frame_indices)
        log_step("5. Extract Frame Batch", t, step_metrics=step_metrics)

        # --- CRITICAL OPERATION 6: TensorFlow Tensor Normalization (GPU) ---
        t = log_step("6. Convert & Normalize GPU Tensor", step_metrics=step_metrics)
        video = tf.cast(frames, tf.float32) / 255.0
        log_step("6. Convert & Normalize GPU Tensor", t, step_metrics=step_metrics)

        # --- CRITICAL OPERATION 7: Apply Transforms ---
        t = log_step("7. Apply Transforms", step_metrics=step_metrics)
        for transform in self.transforms:
            video = transform(video, loading_option)
        log_step("7. Apply Transforms", t, step_metrics=step_metrics)

        return tf.squeeze(video)


if __name__ == "__main__":
    gpus = tf.config.list_physical_devices("GPU")
    gpu_name = gpus[GPU_INDEX].name if gpus else "CPU Only (No GPU detected)"

    print("==========================================")
    print(f"  STARTING GET_RAW_DATA TEST ({NUM_RUNS} RUNS) ")
    print(f"  Target Device: {gpu_name}")
    print("==========================================\n")

    # --- Load Parquet & Get Sample IDs ---
    print(f"Loading dataset from: {WIDE_FILE}")
    df = pd.read_parquet(WIDE_FILE)

    if SAMPLE_ID_COL in df.columns:
        sample_ids = df[SAMPLE_ID_COL].dropna().values
    else:
        sample_ids = df.iloc[:, 0].dropna().values

    tester = DataFetcherTester(gpu_index=GPU_INDEX)
    step_metrics = defaultdict(list)
    total_times = []
    successful_runs = 0

    # --- Run Loop ---
    for run_idx in range(1, NUM_RUNS + 1):
        selected_sample_id = str(np.random.choice(sample_ids))
        print(f"\n--- RUN {run_idx}/{NUM_RUNS} | Sample ID: {selected_sample_id} ---")

        total_start = time.perf_counter()
        try:
            output = tester.get_raw_data(
                selected_sample_id, LOADING_OPTION, step_metrics=step_metrics
            )

            # Ensure GPU operation execution resolves prior to timing calculation
            _ = output.shape

            total_elapsed = (time.perf_counter() - total_start) * 1000
            total_times.append(total_elapsed)
            successful_runs += 1
            print(
                f"Run {run_idx} finished in {total_elapsed:.2f} ms | Output Shape: {output.shape} | Device: {output.device}"
            )

        except Exception as e:
            print(f"[ERROR] Run {run_idx} failed with exception: {e}")

    # --- Final Summary Report ---
    print("\n" + "=" * 66)
    print(f"           BENCHMARK SUMMARY ({successful_runs}/{NUM_RUNS} SUCCESSFUL RUNS)")
    print("=" * 66)

    if successful_runs > 0:
        print(f"{'Step Name':<50} | {'Avg Time (ms)':<12}")
        print("-" * 66)
        for step_name, times in step_metrics.items():
            avg_time = np.mean(times)
            print(f"{step_name:<50} | {avg_time:>10.2f} ms")
        print("-" * 66)
        print(f"{'TOTAL EXECUTION TIME (AVG)':<50} | {np.mean(total_times):>10.2f} ms")
    else:
        print("No successful runs to compute statistics.")
    print("=" * 66)
