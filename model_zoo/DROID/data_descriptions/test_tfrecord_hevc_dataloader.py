import io
import time
from collections import defaultdict
from datetime import datetime
import av
import numpy as np
import tensorflow as tf

# =====================================================================
# CONFIGURATION VARIABLES (Set your test parameters here)
# =====================================================================
# Directory containing the hevc-*.tfrecord shards written by
# code/create_tfrecord_test_datasets.py
TFRECORD_DIR = "/mnt/disks/droid-af/data/tf-record/hevc"

# Number of times to repeat the benchmark
NUM_RUNS = 50

# Shuffle buffer for the TFRecord stream (records are pulled in shuffled
# order, which is the realistic training access pattern for TFRecords)
SHUFFLE_BUFFER = 64

# Keyframe interval the HEVC dataset was encoded with (--gop in the
# creation script). Random start frames are aligned to multiples of this
# so decoding can seek straight to a keyframe instead of decoding from 0.
GOP = 16

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

FEATURE_SPEC = {
    "sample_id": tf.io.FixedLenFeature([], tf.string),
    "study": tf.io.FixedLenFeature([], tf.string),
    "view": tf.io.FixedLenFeature([], tf.string),
    "video": tf.io.FixedLenFeature([], tf.string),
    "codec": tf.io.FixedLenFeature([], tf.string),
    "nframes": tf.io.FixedLenFeature([], tf.int64),
    "height": tf.io.FixedLenFeature([], tf.int64),
    "width": tf.io.FixedLenFeature([], tf.int64),
    "fps": tf.io.FixedLenFeature([], tf.float32),
}


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


def build_record_iterator():
    """Shuffled, repeating stream over all TFRecord shards."""
    files = sorted(tf.io.gfile.glob(f"{TFRECORD_DIR}/*.tfrecord"))
    if not files:
        raise FileNotFoundError(f"No .tfrecord files found in {TFRECORD_DIR}")
    print(f"Found {len(files)} TFRecord shard(s) in {TFRECORD_DIR}")
    ds = tf.data.Dataset.from_tensor_slices(files)
    ds = ds.shuffle(len(files))
    ds = ds.interleave(
        tf.data.TFRecordDataset,
        cycle_length=min(4, len(files)),
        num_parallel_calls=tf.data.experimental.AUTOTUNE,
    )
    ds = ds.shuffle(SHUFFLE_BUFFER)
    ds = ds.repeat()
    return iter(ds)


class DataFetcherTester:

    def __init__(self):
        self.start_frame = START_FRAME
        self.randomize_start_frame = RANDOMIZE_START_FRAME
        self.nframes = NFRAMES
        self.skip_modulo = SKIP_MODULO
        self.gop = GOP
        self.transforms = TRANSFORMS

    def get_raw_data(self, record_iterator, loading_option=None, step_metrics=None):
        # --- CRITICAL OPERATION 1: Fetch Serialized Record from TFRecord Stream ---
        t = log_step("1. Fetch Serialized Record (TFRecord IO)", step_metrics=step_metrics)
        raw_record = next(record_iterator)
        log_step("1. Fetch Serialized Record (TFRecord IO)", t, step_metrics=step_metrics)

        # --- CRITICAL OPERATION 2: Parse tf.train.Example ---
        t = log_step("2. Parse tf.train.Example", step_metrics=step_metrics)
        parsed = tf.io.parse_single_example(raw_record, FEATURE_SPEC)
        sample_id = parsed["sample_id"].numpy().decode("utf-8")
        video_bytes = parsed["video"].numpy()
        total_frames = int(parsed["nframes"].numpy())
        fps = float(parsed["fps"].numpy())
        log_step("2. Parse tf.train.Example", t, step_metrics=step_metrics)

        # --- CRITICAL OPERATION 3: Open PyAV Video Container ---
        t = log_step("3. Open PyAV Video Container", step_metrics=step_metrics)
        in_mem_bytes_io = io.BytesIO(video_bytes)
        video_container = av.open(in_mem_bytes_io, metadata_errors="ignore")
        stream = video_container.streams.video[0]
        stream.thread_type = "AUTO"
        log_step("3. Open PyAV Video Container", t, step_metrics=step_metrics)

        start_frame = self.start_frame

        # --- CRITICAL OPERATION 4: Randomize GOP-Aligned Start Frame ---
        # Start frames are snapped to keyframe boundaries (multiples of GOP)
        # so the seek in step 5 lands exactly on a keyframe and only
        # nframes need decoding — this is the payoff of the HEVC encode.
        if self.randomize_start_frame:
            t = log_step(
                "4. Randomize GOP-Aligned Start Frame", step_metrics=step_metrics
            )
            frame_range = total_frames - (self.nframes * self.skip_modulo)
            if frame_range > 0:
                start_frame = self.gop * np.random.randint(frame_range // self.gop + 1)
            log_step(
                "4. Randomize GOP-Aligned Start Frame", t, step_metrics=step_metrics
            )

        # --- CRITICAL OPERATION 5: Seek to Keyframe ---
        t = log_step("5. Seek to Keyframe", step_metrics=step_metrics)
        if start_frame > 0 and fps > 0:
            # Convert frame index -> stream time_base units; default seek mode
            # lands on the keyframe at/before the requested pts.
            seek_pts = int(round(start_frame / fps / float(stream.time_base)))
            video_container.seek(seek_pts, stream=stream)
        log_step("5. Seek to Keyframe", t, step_metrics=step_metrics)

        # --- CRITICAL OPERATION 6: Frame Decoding & Array Extraction ---
        t = log_step("6. Decode Video Frames", step_metrics=step_metrics)
        frames = []
        window = self.nframes * self.skip_modulo
        i = 0  # frame index relative to the seek point (keyframe == start_frame)
        for frame in video_container.decode(video=0):
            if len(frames) == self.nframes or i >= window:
                break
            if self.skip_modulo > 1:
                if (i % self.skip_modulo) != 0:
                    i += 1
                    continue
            frames.append(frame.to_ndarray(format="rgb24"))
            i += 1
        video_container.close()
        # Pad short clips by cycling the decoded window
        # (mirrors the itertools.cycle padding of the old loader).
        decoded_count = len(frames)
        while frames and len(frames) < self.nframes:
            frames.append(frames[len(frames) % decoded_count])
        log_step("6. Decode Video Frames", t, step_metrics=step_metrics)

        # --- CRITICAL OPERATION 7: NumPy Array Normalization ---
        t = log_step("7. Convert & Normalize NumPy Array", step_metrics=step_metrics)
        video = np.array(frames, dtype="float32") / 255.0
        log_step("7. Convert & Normalize NumPy Array", t, step_metrics=step_metrics)

        # --- CRITICAL OPERATION 8: Apply Transforms ---
        t = log_step("8. Apply Transforms", step_metrics=step_metrics)
        for transform in self.transforms:
            video = transform(video, loading_option)
        log_step("8. Apply Transforms", t, step_metrics=step_metrics)

        return np.squeeze(np.asarray(video, dtype="float32")), sample_id


if __name__ == "__main__":
    print("==========================================")
    print(f"  STARTING TFRECORD HEVC TEST ({NUM_RUNS} RUNS) ")
    print("==========================================\n")

    record_iterator = build_record_iterator()

    tester = DataFetcherTester()
    step_metrics = defaultdict(list)
    total_times = []
    successful_runs = 0

    # --- Run Loop ---
    for run_idx in range(1, NUM_RUNS + 1):
        print(f"\n--- RUN {run_idx}/{NUM_RUNS} ---")

        total_start = time.perf_counter()
        try:
            output, sample_id = tester.get_raw_data(
                record_iterator, LOADING_OPTION, step_metrics=step_metrics
            )
            total_elapsed = (time.perf_counter() - total_start) * 1000
            total_times.append(total_elapsed)
            successful_runs += 1
            print(
                f"Run {run_idx} finished in {total_elapsed:.2f} ms | Sample ID: {sample_id} | Output Shape: {output.shape}"
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
