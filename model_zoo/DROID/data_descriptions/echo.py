import os
import io
import av
import itertools

import lmdb

import numpy as np
import pandas as pd
import tensorflow as tf

from ml4ht.data.data_description import DataDescription

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
