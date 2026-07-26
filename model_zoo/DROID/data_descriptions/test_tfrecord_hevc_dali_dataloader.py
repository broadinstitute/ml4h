import time
from collections import defaultdict
from datetime import datetime
import numpy as np
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
# Directory containing the hevc-*.tfrecord shards written by
# code/create_tfrecord_test_datasets.py
TFRECORD_DIR = "/mnt/disks/droid-af/data/tf-record/hevc"

# GPU Device Index
GPU_INDEX = 0

# Number of times to repeat the benchmark
NUM_RUNS = 10

# Shuffle buffer for the TFRecord stream (records are pulled in shuffled
# order, which is the realistic training access pattern for TFRecords)
SHUFFLE_BUFFER = 64

# Keyframe interval the HEVC dataset was encoded with (--gop in the
# creation script). Random start frames are aligned to multiples of this
# so NVDEC can seek to a keyframe and decode only the requested window.
GOP = 16

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

# Whether this DALI build supports frame selection (start_frame /
# sequence_length / stride) on the video decoder. Detected on first use:
# None = unknown, True/False after the first decode attempt. When supported,
# NVDEC seeks to the GOP-aligned keyframe and decodes ONLY the requested
# window; otherwise the whole clip is decoded and sliced on GPU afterwards.
FRAME_SELECTION = {"supported": None}

# One-time warning flag for the DLPack -> host-memory fallback path.
FALLBACK_WARNED = {"done": False}


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


def run_dali_decode(video_array, gpu_index, start_frame=None, nframes=None, stride=1):
    """Decode an in-memory encoded video on GPU. If start_frame is given,
    request only that window from the decoder (needs DALI with frame
    selection support on the video decoder)."""
    decoder_kwargs = {}
    if start_frame is not None:
        decoder_kwargs = dict(
            start_frame=start_frame,
            sequence_length=nframes,
            stride=stride,
        )

    @pipeline_def(batch_size=1, num_threads=2, device_id=gpu_index)
    def video_decode_pipeline():
        # batch=False tells DALI video_array is a single sample, not a batch of scalars
        encoded = fn.external_source(source=[video_array], batch=False, dtype=types.UINT8)
        decoded_video = dali_decode_video(encoded, device="mixed", **decoder_kwargs)
        return decoded_video

    pipe = video_decode_pipeline()
    pipe.build()
    pipe_output = pipe.run()
    # Extract DALI output tensor (Shape: [F, H, W, C])
    return pipe_output[0][0]


class DataFetcherTester:

    def __init__(self, gpu_index=0):
        self.start_frame = START_FRAME
        self.randomize_start_frame = RANDOMIZE_START_FRAME
        self.nframes = NFRAMES
        self.skip_modulo = SKIP_MODULO
        self.gop = GOP
        self.transforms = TRANSFORMS
        self.gpu_index = gpu_index

    def get_raw_data(self, record_iterator, loading_option=None, step_metrics=None):
        # --- CRITICAL OPERATION 1: Fetch Serialized Record from TFRecord Stream ---
        t = log_step("1. Fetch Serialized Record (TFRecord IO)", step_metrics=step_metrics)
        raw_record = next(record_iterator)
        log_step("1. Fetch Serialized Record (TFRecord IO)", t, step_metrics=step_metrics)

        # --- CRITICAL OPERATION 2: Parse tf.train.Example ---
        t = log_step("2. Parse tf.train.Example", step_metrics=step_metrics)
        parsed = tf.io.parse_single_example(raw_record, FEATURE_SPEC)
        sample_id = parsed["sample_id"].numpy().decode("utf-8")
        video_array = np.frombuffer(parsed["video"].numpy(), dtype=np.uint8)
        total_frames = int(parsed["nframes"].numpy())
        log_step("2. Parse tf.train.Example", t, step_metrics=step_metrics)

        # --- CRITICAL OPERATION 3: Randomize GOP-Aligned Start Frame ---
        # Uses TFRecord metadata (no probe decode). Start frames snap to
        # keyframe boundaries so NVDEC seeks land exactly on a keyframe.
        t = log_step("3. Randomize GOP-Aligned Start Frame", step_metrics=step_metrics)
        start_frame = self.start_frame
        window = self.nframes * self.skip_modulo
        if self.randomize_start_frame:
            frame_range = total_frames - window
            if frame_range > 0:
                start_frame = self.gop * np.random.randint(frame_range // self.gop + 1)
        log_step("3. Randomize GOP-Aligned Start Frame", t, step_metrics=step_metrics)

        # --- CRITICAL OPERATION 4: Decode Video via DALI GPU (NVDEC) ---
        t = log_step("4. Decode Video via DALI GPU (NVDEC)", step_metrics=step_metrics)
        window_decoded = False
        window_fits = start_frame + window <= total_frames
        if FRAME_SELECTION["supported"] is not False and window_fits:
            try:
                dali_tensor = run_dali_decode(
                    video_array, self.gpu_index,
                    start_frame=start_frame, nframes=self.nframes, stride=self.skip_modulo,
                )
                FRAME_SELECTION["supported"] = True
                window_decoded = True
            except Exception as e:
                if FRAME_SELECTION["supported"] is None:
                    print(
                        f"[WARN] DALI frame selection unavailable ({e}); "
                        "falling back to full-clip decode + GPU slice."
                    )
                FRAME_SELECTION["supported"] = False
        if not window_decoded:
            dali_tensor = run_dali_decode(video_array, self.gpu_index)

        # Convert DALI GPU tensor directly to TensorFlow GPU Tensor via DLPack.
        # Older DALI exposed .as_dlpack(); newer versions (e.g. 1.50) implement
        # the standard __dlpack__ protocol instead — try both.
        try:
            if hasattr(dali_tensor, "as_dlpack"):
                capsule = dali_tensor.as_dlpack()
            else:
                capsule = dali_tensor.__dlpack__()
            raw_frames = tf.experimental.dlpack.from_dlpack(capsule)
        except Exception as e:
            if not FALLBACK_WARNED["done"]:
                print(
                    f"[WARN] DLPack GPU handoff failed ({type(dali_tensor).__name__}: {e}); "
                    "copying through host memory. If the tensor type is TensorCPU, the "
                    "decode itself ran on CPU, not NVDEC."
                )
                FALLBACK_WARNED["done"] = True
            cpu_tensor = dali_tensor.as_cpu() if hasattr(dali_tensor, "as_cpu") else dali_tensor
            # np.array() uses the tensor's array interface; Tensor*.as_array()
            # was removed in newer DALI (only TensorLists have it).
            raw_frames = tf.constant(np.array(cpu_tensor))
        log_step("4. Decode Video via DALI GPU (NVDEC)", t, step_metrics=step_metrics)

        # --- CRITICAL OPERATION 5: Extract Frame Batch (GPU) ---
        t = log_step("5. Extract Frame Batch (GPU)", step_metrics=step_metrics)
        if window_decoded:
            # Decoder already returned exactly the requested window.
            frames = raw_frames
        else:
            decoded_frames = int(raw_frames.shape[0])
            end_frame = start_frame + window
            # Cycle indices past the end so short clips are padded by
            # repeating (mirrors the itertools.cycle padding of the old loader).
            frame_indices = [
                idx % decoded_frames
                for idx in range(start_frame, end_frame, self.skip_modulo)
            ]
            frames = tf.gather(raw_frames, frame_indices)
        log_step("5. Extract Frame Batch (GPU)", t, step_metrics=step_metrics)

        # --- CRITICAL OPERATION 6: TensorFlow Tensor Normalization (GPU) ---
        t = log_step("6. Convert & Normalize GPU Tensor", step_metrics=step_metrics)
        video = tf.cast(frames, tf.float32) / 255.0
        log_step("6. Convert & Normalize GPU Tensor", t, step_metrics=step_metrics)

        # --- CRITICAL OPERATION 7: Apply Transforms ---
        t = log_step("7. Apply Transforms", step_metrics=step_metrics)
        for transform in self.transforms:
            video = transform(video, loading_option)
        log_step("7. Apply Transforms", t, step_metrics=step_metrics)

        return tf.squeeze(video), sample_id


if __name__ == "__main__":
    gpus = tf.config.list_physical_devices("GPU")
    gpu_name = gpus[GPU_INDEX].name if gpus else "CPU Only (No GPU detected)"

    print("==========================================")
    print(f"  STARTING TFRECORD HEVC + DALI TEST ({NUM_RUNS} RUNS) ")
    print(f"  Target Device: {gpu_name}")
    print("==========================================\n")

    record_iterator = build_record_iterator()

    tester = DataFetcherTester(gpu_index=GPU_INDEX)
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

            # Ensure GPU operation execution resolves prior to timing calculation
            _ = output.shape

            total_elapsed = (time.perf_counter() - total_start) * 1000
            total_times.append(total_elapsed)
            successful_runs += 1
            print(
                f"Run {run_idx} finished in {total_elapsed:.2f} ms | Sample ID: {sample_id} | "
                f"Output Shape: {output.shape} | Device: {output.device}"
            )

        except Exception as e:
            print(f"[ERROR] Run {run_idx} failed with exception: {e}")

    # --- Final Summary Report ---
    print("\n" + "=" * 66)
    print(f"           BENCHMARK SUMMARY ({successful_runs}/{NUM_RUNS} SUCCESSFUL RUNS)")
    print("=" * 66)

    if successful_runs > 0:
        mode = (
            "NVDEC window decode (frame selection)"
            if FRAME_SELECTION["supported"]
            else "full-clip decode + GPU slice"
        )
        print(f"Decode mode: {mode}")
        print("-" * 66)
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
