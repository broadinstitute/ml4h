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


def _precompute_labels(sample_ids, output_dd, output_dims):
    """Look every sample's labels up once at dataset-build time.

    output_dd.get_raw_data is a pandas lookup; calling it per sample per epoch
    from inside the input pipeline serializes on the GIL and steals time from
    video decoding. Returns one (n_samples, dim) float32 array per output
    head, row-aligned with `sample_ids`, so the pipeline can fetch labels with
    a graph-side gather.
    """
    heads = [np.empty((len(sample_ids), dim), dtype=np.float32) for dim in output_dims]
    for row, sample_id in enumerate(sample_ids):
        labels = output_dd.get_raw_data(sample_id)
        if not isinstance(labels, (list, tuple)):
            labels = [labels]
        for head, label in zip(heads, labels):
            head[row] = np.asarray(label, dtype=np.float32).reshape(-1)
    return heads


def _label_row_table(sample_ids):
    """StaticHashTable mapping sample_id -> row in the precomputed label
    arrays; -1 (the default) marks records that should be filtered out."""
    return tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(
            tf.constant(sample_ids),
            tf.range(len(sample_ids), dtype=tf.int32),
        ),
        default_value=-1,
    )


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
    `sample_ids`, and decodes the video bytes with PyAV. Labels are looked up
    through `output_dd` (EcholabDataDescription) once at build time and served
    from constant tensors, so the per-record py_function does nothing but the
    video decode. `output_dims` lists the per-sample length of each model
    output (regression, classification heads, then survival heads), in the
    order output_dd returns them.

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

    label_consts = [tf.constant(head) for head in _precompute_labels(sample_ids, output_dd, output_dims)]
    row_table = _label_row_table(sample_ids)

    feature_spec = {
        'sample_id': tf.io.FixedLenFeature([], tf.string),
        'video': tf.io.FixedLenFeature([], tf.string),
    }

    def parse(serialized):
        return tf.io.parse_single_example(serialized, feature_spec)

    def keep(parsed):
        return row_table.lookup(parsed['sample_id']) >= 0

    # do_not_convert: tf.py_function runs its callable through AutoGraph by
    # default, which breaks the eager-only tensor control flow in the video
    # transforms (see transforms.py docstring); run as plain eager Python.
    @tf.autograph.experimental.do_not_convert
    def load(video_bytes):
        video = decode_video_clip(
            video_bytes.numpy(),
            n_input_frames,
            skip_modulo=skip_modulo,
            randomize_start_frame=randomize_start_frame,
        )
        for transform in transforms:
            video = transform(video, None)
        return np.asarray(video, dtype=np.float32)

    def to_model_io(parsed):
        video = tf.py_function(load, [parsed['video']], Tout=tf.float32)
        video.set_shape((n_input_frames, 224, 224, 3))
        row = row_table.lookup(parsed['sample_id'])
        labels = [tf.gather(label_const, row) for label_const in label_consts]
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

    The stages are pipelined so no single thread serializes the epoch:
    TFRecord reading/proto parsing/id filtering run inside a parallel tf.data
    graph; a DALI pipeline with a pipelined async executor pulls encoded
    batches through an external_source callback and decodes on NVDEC
    (start_frame / sequence_length / stride, queue depth 2, so batch N+1
    decodes while batch N is post-processed on the host); decoded frames cross
    back as uint8 (4x fewer bytes than float32) into tf.data, where [0, 1]
    normalization, the `transforms` (the same eager TF transforms as the CPU
    path, each self-gated on its own p) and the label gather run in a parallel
    map that overlaps decode and training. Keeping the DALI graph decode-only
    keeps its GPU footprint to the decoded uint8 batch instead of a chain of
    float32 intermediates. Labels are precomputed from `output_dd` once at
    build time and gathered graph-side.

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
    transforms = transforms or []
    output_dims = list(output_dims or [])
    if not output_dims:
        raise ValueError('build_echo_dali_dataset needs at least one entry in output_dims')
    n_outputs = len(output_dims)

    files = sorted(tf.io.gfile.glob(os.path.join(tfrecord_dir, '*.tfrecord')))
    if not files:
        raise FileNotFoundError(f'No .tfrecord shards found in {tfrecord_dir}')

    label_consts = [tf.constant(head) for head in _precompute_labels(sample_ids, output_dd, output_dims)]
    row_table = _label_row_table(sample_ids)
    row_index = {sample_id: row for row, sample_id in enumerate(sample_ids)}

    feature_spec = {
        'sample_id': tf.io.FixedLenFeature([], tf.string),
        'video': tf.io.FixedLenFeature([], tf.string),
        'nframes': tf.io.FixedLenFeature([], tf.int64),
    }

    def parse(serialized):
        return tf.io.parse_single_example(serialized, feature_spec)

    def keep(parsed):
        return row_table.lookup(parsed['sample_id']) >= 0

    def record_stream():
        # Reading, proto parsing and id filtering run in parallel inside
        # tf.data (C++), replacing the serial Example.FromString loop; records
        # not in sample_ids are dropped graph-side, before their video bytes
        # are ever copied out to Python.
        records = tf.data.TFRecordDataset(files, num_parallel_reads=tf.data.AUTOTUNE)
        records = records.map(parse, num_parallel_calls=tf.data.AUTOTUNE)
        records = records.filter(keep).prefetch(tf.data.AUTOTUNE)
        for parsed in records.as_numpy_iterator():
            yield (
                parsed['sample_id'].decode('utf-8'),
                parsed['video'],
                int(parsed['nframes']),
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

    def batch_stream():
        batch_rows, batch_encoded, batch_starts = [], [], []
        for sample_id, video_bytes, nframes in shuffled_stream():
            start = 0
            if randomize_start_frame:
                frame_range = nframes - n_input_frames * skip_modulo
                if frame_range > 0:
                    start = int(np.random.randint(frame_range))
            batch_rows.append(row_index[sample_id])
            batch_encoded.append(np.frombuffer(video_bytes, dtype=np.uint8))
            batch_starts.append(start)
            if len(batch_rows) == batch_size:
                yield (
                    batch_encoded,
                    np.array(batch_starts, dtype=np.int32),
                    np.array(batch_rows, dtype=np.int32),
                )
                batch_rows, batch_encoded, batch_starts = [], [], []
        # Any final partial batch is dropped, matching drop_remainder=True.

    # The current epoch's batch stream lives in a mutable cell so the
    # external_source callback (bound once at pipeline build) can be
    # re-pointed at a fresh pass each epoch; its StopIteration ends the DALI
    # epoch and pipe.reset() re-arms the pipeline for the next one.
    stream_cell = [None]

    def dali_source():
        if stream_cell[0] is None:
            raise StopIteration
        return next(stream_cell[0])

    # Pipelined async executor with queue depth 2: DALI pulls encoded batches
    # through the callback and NVDEC decodes batch N+1 while the host
    # normalizes/augments batch N. row_idx rides through the pipeline so each
    # decoded batch stays aligned with its rows in the label arrays.
    @dali_pipeline_def(batch_size=batch_size, num_threads=num_threads, device_id=device_id,
                       prefetch_queue_depth=2)
    def video_pipe():
        encoded, start_frame, row_idx = dali_fn.external_source(
            source=dali_source,
            num_outputs=3,
            batch=True,
            dtype=[dali_types.UINT8, dali_types.INT32, dali_types.INT32],
        )
        video = dali_fn.experimental.decoders.video(
            encoded,
            device='mixed',
            start_frame=start_frame,
            sequence_length=n_input_frames,
            stride=skip_modulo,
            pad_mode='edge',
        )
        return video, row_idx

    pipe = video_pipe()
    pipe.build()

    def decoded_batches():
        stream_cell[0] = batch_stream()
        pipe.reset()
        try:
            while True:
                try:
                    videos, row_idx = pipe.run()
                except StopIteration:
                    break
                # (B, T, H, W, 3) uint8 decoded on the GPU; stays uint8 so only
                # a quarter of the float32 bytes cross PCIe and the generator
                # boundary. Normalization happens in the tf.data map below.
                yield videos.as_cpu().as_array(), row_idx.as_array()
        finally:
            stream_cell[0] = None

    # do_not_convert for the same reason as the CPU path: the transforms need
    # eager tensor control flow. Each transform still gates itself on its own
    # p, with parameters shared across a clip's frames but not across clips.
    @tf.autograph.experimental.do_not_convert
    def augment_batch(videos):
        videos = videos.numpy().astype(np.float32) / 255.0
        clips = []
        for clip in videos:
            for transform in transforms:
                clip = transform(clip, None)
            clips.append(np.asarray(clip, dtype=np.float32))
        return np.stack(clips)

    def to_model_io(videos, row_idx):
        if transforms:
            videos = tf.py_function(augment_batch, [videos], Tout=tf.float32)
        else:
            videos = tf.cast(videos, tf.float32) / 255.0
        videos.set_shape((batch_size, n_input_frames, 224, 224, 3))
        labels = [tf.gather(label_const, row_idx) for label_const in label_consts]
        if n_outputs == 1:
            return videos, labels[0]
        return videos, tuple(labels)

    dataset = tf.data.Dataset.from_generator(
        decoded_batches,
        output_signature=(
            tf.TensorSpec(shape=(batch_size, n_input_frames, 224, 224, 3), dtype=tf.uint8),
            tf.TensorSpec(shape=(batch_size,), dtype=tf.int32),
        ),
    )
    return dataset.map(to_model_io, num_parallel_calls=tf.data.AUTOTUNE)
