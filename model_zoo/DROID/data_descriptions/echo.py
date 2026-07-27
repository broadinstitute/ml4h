import os
import io
import av
import itertools

import lmdb

import numpy as np
import pandas as pd
import tensorflow as tf

from ml4ht.data.data_description import DataDescription

try:
    from nvidia.dali import fn as dali_fn
    from nvidia.dali import math as dali_math
    from nvidia.dali import pipeline_def as dali_pipeline_def
    from nvidia.dali import types as dali_types
    DALI_AVAILABLE = True
except ImportError:
    DALI_AVAILABLE = False

VIEW_OPTION_KEY = 'view'

metadata_elements = [
    'PhotometricInterpretation',
    'TransferSyntaxUID',
    'SamplesPerPixel',
    'BitsAllocated',
    'BitsStored',
    'HighBit',
    'PixelRepresentation',
    'PlanarConfiguration',
    'NumberOfFrames',
    'Rows',
    'Columns',
]


class LmdbEchoStudyVideoDataDescription(DataDescription):

    def __init__(
            self,
            local_lmdb_dir: str,
            name: str,
            transforms=None,
            nframes: int = None,
            skip_modulo: int = 1,
            start_frame=0,
            randomize_start_frame = False,
    ):

        self.local_lmdb_dir = local_lmdb_dir
        self._name = name
        self.start_frame = start_frame
        self.nframes = nframes
        # transformations
        self.transforms = transforms or []
        self.skip_modulo = skip_modulo
        self.randomize_start_frame = randomize_start_frame

    def get_loading_options(self, sample_id):
        _, study, view = sample_id.split('_')
        lmdb_folder = os.path.join(self.local_lmdb_dir, f"{study}.lmdb")
        lmdb_log = pd.read_parquet(os.path.join(lmdb_folder, f'log_{study}.pq')).set_index('view')
        lmdb_log = lmdb_log[lmdb_log['stored']]

        if view not in lmdb_log.index:
            raise ValueError('View not saved in the LMDB')

        return [
            {VIEW_OPTION_KEY: view},
        ]

    def get_raw_data(self, sample_id, loading_option=None):
        try:
            sample_id = sample_id.decode('UTF-8')
        except (UnicodeDecodeError, AttributeError):
            pass
        _, study, view = sample_id.split('_')

        lmdb_folder = os.path.join(self.local_lmdb_dir, f"{study}.lmdb")

        env = lmdb.open(lmdb_folder, readonly=True, lock=False)

        frames = []
        with env.begin(buffers=True) as txn:
            in_mem_bytes_io = io.BytesIO(txn.get(view.encode('utf-8')))
            video_container = av.open(in_mem_bytes_io, metadata_errors="ignore")
            # Multithreaded frame decode (MJPG is intra-only, so this is safe/deterministic).
            stream = video_container.streams.video[0]
            stream.thread_type = "AUTO"

            # Use a local start_frame so the shared instance isn't mutated (safe under
            # parallel decoding). Only compute the total frame count when we actually
            # need it for randomization.
            start_frame = self.start_frame
            if self.randomize_start_frame:
                # Prefer the container's frame count from metadata (no decode). Fall
                # back to a counting decode pass only when metadata is missing/zero.
                total_frames = stream.frames
                if not total_frames:
                    total_frames = sum(1 for _ in video_container.decode(video=0))
                    video_container.seek(0)
                frame_range = total_frames - (self.nframes * self.skip_modulo)
                if frame_range > 0:
                    start_frame = np.random.randint(frame_range)

            # itertools.cycle loops the decoded frames so clips shorter than the
            # requested window are padded by repeating, preserving prior behavior.
            video_frames = itertools.cycle(video_container.decode(video=0))
            for i, frame in enumerate(video_frames):
                if len(frames) == self.nframes:
                    break
                if i < start_frame:
                    continue
                if self.skip_modulo > 1:
                    if ((i - start_frame) % self.skip_modulo) != 0:
                        continue
                # Decode straight to an RGB ndarray, skipping the PyAV -> PIL -> ndarray round-trip.
                frame = frame.to_ndarray(format='rgb24')
                frames.append(frame)
            del video_frames
            video_container.close()
        env.close()

        # Transforms operate on the whole normalized clip of shape (T, H, W, C) in
        # [0, 1] so that augmentations requiring cross-frame consistency (jitter,
        # rotation, flips, mask sector) can share sampled parameters across frames.
        # See data_descriptions/transforms.py for the transform contract.
        video = np.array(frames, dtype='float32') / 255.
        for transform in self.transforms:
            video = transform(video, loading_option)
        return np.squeeze(np.asarray(video, dtype='float32'))

    @property
    def name(self):
        return self._name


def decode_video_clip(video_bytes, nframes, skip_modulo=1, start_frame=0, randomize_start_frame=False):
    """Decode in-memory video bytes (H.264/MJPEG/...) to a (nframes, H, W, 3) float32 clip in [0, 1].

    Mirrors the frame selection of LmdbEchoStudyVideoDataDescription: optional
    random start frame, every skip_modulo-th frame, and clips shorter than the
    requested window are padded by cycling from the beginning.
    """
    with av.open(io.BytesIO(video_bytes), metadata_errors='ignore') as video_container:
        stream = video_container.streams.video[0]
        stream.thread_type = 'AUTO'

        if randomize_start_frame:
            total_frames = stream.frames
            if not total_frames:
                total_frames = sum(1 for _ in video_container.decode(video=0))
                video_container.seek(0)
            frame_range = total_frames - (nframes * skip_modulo)
            if frame_range > 0:
                start_frame = np.random.randint(frame_range)

        frames = []
        video_frames = itertools.cycle(video_container.decode(video=0))
        for i, frame in enumerate(video_frames):
            if len(frames) == nframes:
                break
            if i < start_frame:
                continue
            if skip_modulo > 1 and ((i - start_frame) % skip_modulo) != 0:
                continue
            frames.append(frame.to_ndarray(format='rgb24'))
        del video_frames

    return np.array(frames, dtype='float32') / 255.


def build_echo_tfrecord_dataset(
        tfrecord_dir,
        sample_ids,
        output_dd,
        batch_size,
        n_input_frames,
        skip_modulo=1,
        output_dims=None,
        transforms=None,
        randomize_start_frame=False,
        shuffle=False,
        shuffle_buffer=256,
):
    """tf.data pipeline over a sharded TFRecord echo dataset.

    Replaces the LMDB generator pipeline: reads serialized examples written by
    create_tfrecord_a4c_dataset.py, keeps only records whose sample_id is in
    `sample_ids`, decodes the video bytes with PyAV, and looks the labels up
    through `output_dd` (EcholabDataDescription). `output_dims` lists the
    per-sample length of each model output (regression, classification heads,
    then survival heads), in the order output_dd returns them.

    Returns a finite dataset of (video, labels) batches with
    drop_remainder=True; the training call site adds .repeat().
    """
    sample_ids = [s.decode('utf-8') if isinstance(s, bytes) else str(s) for s in sample_ids]
    if not sample_ids:
        raise ValueError('build_echo_tfrecord_dataset called with no sample_ids')
    transforms = transforms or []
    output_dims = list(output_dims or [])
    if not output_dims:
        raise ValueError('build_echo_tfrecord_dataset needs at least one entry in output_dims')
    n_outputs = len(output_dims)

    files = sorted(tf.io.gfile.glob(os.path.join(tfrecord_dir, '*.tfrecord')))
    if not files:
        raise FileNotFoundError(f'No .tfrecord shards found in {tfrecord_dir}')

    id_table = tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(
            tf.constant(sample_ids),
            tf.ones(len(sample_ids), dtype=tf.int32),
        ),
        default_value=0,
    )

    feature_spec = {
        'sample_id': tf.io.FixedLenFeature([], tf.string),
        'video': tf.io.FixedLenFeature([], tf.string),
    }

    def parse(serialized):
        return tf.io.parse_single_example(serialized, feature_spec)

    def keep(parsed):
        return id_table.lookup(parsed['sample_id']) > 0

    # do_not_convert: tf.py_function runs its callable through AutoGraph by
    # default, which breaks the eager-only tensor control flow in the video
    # transforms (see transforms.py docstring); run as plain eager Python.
    @tf.autograph.experimental.do_not_convert
    def load(sample_id, video_bytes):
        sample_id = sample_id.numpy().decode('utf-8')
        video = decode_video_clip(
            video_bytes.numpy(),
            n_input_frames,
            skip_modulo=skip_modulo,
            randomize_start_frame=randomize_start_frame,
        )
        for transform in transforms:
            video = transform(video, None)
        labels = output_dd.get_raw_data(sample_id)
        if not isinstance(labels, (list, tuple)):
            labels = [labels]
        return [np.asarray(video, dtype=np.float32)] + [np.asarray(label, dtype=np.float32) for label in labels]

    def to_model_io(parsed):
        outputs = tf.py_function(
            load,
            [parsed['sample_id'], parsed['video']],
            Tout=[tf.float32] * (1 + n_outputs),
        )
        video = outputs[0]
        video.set_shape((n_input_frames, 224, 224, 3))
        labels = outputs[1:]
        for label, dim in zip(labels, output_dims):
            label.set_shape((dim,))
        if n_outputs == 1:
            return video, labels[0]
        return video, tuple(labels)

    dataset = tf.data.TFRecordDataset(files, num_parallel_reads=tf.data.AUTOTUNE)
    dataset = dataset.map(parse, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.filter(keep)
    if shuffle:
        dataset = dataset.shuffle(shuffle_buffer, reshuffle_each_iteration=True)
    dataset = dataset.map(to_model_io, num_parallel_calls=tf.data.AUTOTUNE)
    return dataset.batch(batch_size, drop_remainder=True)


def _dali_augment(video, transform):
    """Return `video` with the DALI-native equivalent of `transform` applied.

    These are not ports of the TF transforms in transforms.py — each one is
    simply the closest built-in DALI operator, configured from the transform's
    parameters and applied to the whole clip (FHWC sequence) on the GPU.
    """
    name = type(transform).__name__
    if name == 'RandomJitterRotate':
        angle_deg = transform.max_angle * 180.0 / np.pi
        angle = dali_fn.random.uniform(range=[-angle_deg, angle_deg])
        # Frames are 224x224 (enforced by the dataset signature).
        offset = dali_fn.random.uniform(range=[-transform.max_shift, transform.max_shift], shape=[2]) * 224.0
        matrix = dali_fn.transforms.rotation(angle=angle, center=[112.0, 112.0])
        matrix = dali_fn.transforms.translation(matrix, offset=offset)
        return dali_fn.warp_affine(
            video, matrix=matrix, fill_value=0.0, interp_type=dali_types.INTERP_LINEAR,
        )
    if name == 'RandomSectorMask':
        size = dali_fn.random.uniform(range=[0.0, transform.max_frac], shape=[2])
        anchor = dali_fn.random.uniform(range=[0.0, 1.0], shape=[2]) * (1.0 - size)
        return dali_fn.erase(
            video, anchor=anchor, shape=size,
            normalized_anchor=True, normalized_shape=True,
            axis_names='HW', fill_value=0.0,
        )
    if name == 'RandomFlip':
        return dali_fn.flip(
            video,
            horizontal=dali_fn.random.coin_flip(probability=transform.horizontal_prob),
            vertical=dali_fn.random.coin_flip(probability=transform.vertical_prob),
        )
    if name == 'RandomGaussianNoise':
        stddev = dali_fn.random.uniform(range=[0.0, transform.max_fraction])
        return dali_math.clamp(dali_fn.noise.gaussian(video, stddev=stddev), 0.0, 1.0)
    if name == 'RandomBrightnessContrast':
        shift = dali_fn.random.normal(mean=0.0, stddev=transform.std) \
            * dali_fn.random.coin_flip(probability=transform.brightness_prob, dtype=dali_types.FLOAT)
        contrast = 1.0 + dali_fn.random.normal(mean=0.0, stddev=transform.std) \
            * dali_fn.random.coin_flip(probability=transform.contrast_prob, dtype=dali_types.FLOAT)
        return dali_math.clamp(
            dali_fn.brightness_contrast(video, brightness_shift=shift, contrast=contrast, contrast_center=0.5),
            0.0, 1.0,
        )
    raise ValueError(f'No DALI equivalent for transform {name}')


def _dali_gate(video, augmented, p):
    """Per-sample blend that applies `augmented` with probability `p`."""
    if p >= 1.0:
        return augmented
    flag = dali_fn.random.coin_flip(probability=p, dtype=dali_types.FLOAT).gpu()
    return flag * augmented + (1.0 - flag) * video


def build_echo_dali_dataset(
        tfrecord_dir,
        sample_ids,
        output_dd,
        batch_size,
        n_input_frames,
        skip_modulo=1,
        output_dims=None,
        transforms=None,
        randomize_start_frame=False,
        shuffle=False,
        shuffle_buffer=256,
        device_id=0,
        num_threads=4,
):
    """Same contract as build_echo_tfrecord_dataset, but the H.264 decode runs
    on the GPU with NVIDIA DALI (NVDEC).

    A Python generator streams serialized examples out of the TFRecord shards,
    filters by sample_id and shuffles, then feeds each batch of encoded video
    bytes to a DALI pipeline whose experimental video decoder (device='mixed')
    decodes on NVDEC and does the frame selection (start_frame /
    sequence_length / stride) and [0, 1] normalization on the GPU. Labels come
    from `output_dd` exactly as before. Each entry in `transforms` is applied
    inside the pipeline as its DALI-native equivalent (see _dali_augment), so
    augmentation also runs on the GPU rather than through the TF transforms.

    One behavioral difference: clips shorter than the requested window are
    padded by repeating the last frame (DALI pad_mode='edge') instead of
    cycling from the beginning.
    """
    if not DALI_AVAILABLE:
        raise ImportError(
            'nvidia.dali is not installed; pip install nvidia-dali-cuda120 '
            'or use build_echo_tfrecord_dataset instead.'
        )
    sample_ids = [s.decode('utf-8') if isinstance(s, bytes) else str(s) for s in sample_ids]
    if not sample_ids:
        raise ValueError('build_echo_dali_dataset called with no sample_ids')
    id_set = set(sample_ids)
    transforms = transforms or []
    output_dims = list(output_dims or [])
    if not output_dims:
        raise ValueError('build_echo_dali_dataset needs at least one entry in output_dims')

    files = sorted(tf.io.gfile.glob(os.path.join(tfrecord_dir, '*.tfrecord')))
    if not files:
        raise FileNotFoundError(f'No .tfrecord shards found in {tfrecord_dir}')

    # Synchronous executor (no pipelining/async) so each batch can be pushed
    # with feed_input and pulled with run from the generator thread.
    @dali_pipeline_def(batch_size=batch_size, num_threads=num_threads, device_id=device_id,
                       exec_pipelined=False, exec_async=False, prefetch_queue_depth=1)
    def video_pipe():
        encoded = dali_fn.external_source(name='encoded', dtype=dali_types.UINT8)
        start_frame = dali_fn.external_source(name='start_frame', dtype=dali_types.INT32)
        frames = dali_fn.experimental.decoders.video(
            encoded,
            device='mixed',
            start_frame=start_frame,
            sequence_length=n_input_frames,
            stride=skip_modulo,
            pad_mode='edge',
        )
        video = dali_fn.cast(frames, dtype=dali_types.FLOAT) / 255.0
        for transform in transforms:
            video = _dali_gate(video, _dali_augment(video, transform), transform.p)
        return video

    pipe = video_pipe()
    pipe.build()

    def record_stream():
        for raw in tf.data.TFRecordDataset(files):
            example = tf.train.Example.FromString(raw.numpy())
            feats = example.features.feature
            sample_id = feats['sample_id'].bytes_list.value[0].decode('utf-8')
            if sample_id not in id_set:
                continue
            yield (
                sample_id,
                feats['video'].bytes_list.value[0],
                int(feats['nframes'].int64_list.value[0]),
            )

    def shuffled_stream():
        if not shuffle:
            yield from record_stream()
            return
        # Streaming shuffle buffer with the same semantics as tf.data's shuffle.
        buffer = []
        for item in record_stream():
            buffer.append(item)
            if len(buffer) >= shuffle_buffer:
                idx = np.random.randint(len(buffer))
                buffer[idx], buffer[-1] = buffer[-1], buffer[idx]
                yield buffer.pop()
        np.random.shuffle(buffer)
        yield from buffer

    def run_batch(batch_ids, batch_encoded, batch_starts):
        pipe.feed_input('encoded', batch_encoded)
        pipe.feed_input('start_frame', np.array(batch_starts, dtype=np.int32))
        (videos,) = pipe.run()
        # (B, T, H, W, 3) float32 in [0, 1], decoded and augmented on the GPU.
        batch_inputs = videos.as_cpu().as_array()

        batch_outputs = [output_dd.get_raw_data(sample_id) for sample_id in batch_ids]
        if isinstance(batch_outputs[0], (list, tuple)):
            batch_outputs = tuple(
                np.stack([sample_output[output_idx] for sample_output in batch_outputs]).astype(np.float32, copy=False)
                for output_idx in range(len(batch_outputs[0]))
            )
        else:
            batch_outputs = np.stack(batch_outputs).astype(np.float32, copy=False)
        return batch_inputs, batch_outputs

    def generator():
        batch_ids, batch_encoded, batch_starts = [], [], []
        for sample_id, video_bytes, nframes in shuffled_stream():
            start = 0
            if randomize_start_frame:
                frame_range = nframes - n_input_frames * skip_modulo
                if frame_range > 0:
                    start = int(np.random.randint(frame_range))
            batch_ids.append(sample_id)
            batch_encoded.append(np.frombuffer(video_bytes, dtype=np.uint8))
            batch_starts.append(start)
            if len(batch_ids) < batch_size:
                continue
            yield run_batch(batch_ids, batch_encoded, batch_starts)
            batch_ids, batch_encoded, batch_starts = [], [], []
        # Any final partial batch is dropped, matching drop_remainder=True.

    video_spec = tf.TensorSpec(shape=(batch_size, n_input_frames, 224, 224, 3), dtype=tf.float32)
    label_specs = [tf.TensorSpec(shape=(batch_size, dim), dtype=tf.float32) for dim in output_dims]
    output_signature = (
        video_spec,
        tuple(label_specs) if len(label_specs) > 1 else label_specs[0],
    )
    return tf.data.Dataset.from_generator(generator, output_signature=output_signature)
