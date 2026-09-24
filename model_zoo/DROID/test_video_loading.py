"""Synthetic integration tests: frame equivalence, finite datasets, and resources."""

from concurrent.futures import ThreadPoolExecutor
import gc
from pathlib import Path
import sys

import av
import lmdb
import numpy as np
import pandas as pd
import psutil
import pytest
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).resolve().parent))
from benchmark_reference import ReferenceVideoDescription
from data_descriptions.echo import LmdbEchoStudyVideoDataDescription
from data_descriptions.echo_dataset import make_dataset, make_inference_dataset
from data_descriptions.video_io import decode_clip, read_video_bytes
from data_descriptions.wide_file import EcholabDataDescription
from data_descriptions.transforms import RandomFlip
from synthetic_lmdb import generate, make_avi


@pytest.fixture(scope='module')
def data(tmp_path_factory):
    root = tmp_path_factory.mktemp('echo') / 'synthetic'
    rows = generate(root, studies=3, views=3, frames=17, size=32)
    return root, [row['sample_id'] for row in rows]


@pytest.mark.parametrize('frames,stride,start', [(1, 1, 0), (8, 1, 0), (12, 3, 4), (32, 4, 31)])
def test_matches_current_loader(data, frames, stride, start):
    root, ids = data
    options = dict(nframes=frames, skip_modulo=stride, start_frame=start)
    reference = ReferenceVideoDescription(str(root / 'lmdb'), 'ref', **options)
    optimized = LmdbEchoStudyVideoDataDescription(str(root / 'lmdb'), 'fast', **options)
    expected = reference.get_raw_data(ids[0]).reshape(frames, 32, 32, 3)
    actual = optimized.get_raw_data(ids[0].encode())
    np.testing.assert_array_equal(actual, expected)
    assert actual.dtype == np.float32


def test_random_crop_matches_current_loader(data):
    root, ids = data
    options = dict(nframes=3, skip_modulo=2, randomize_start_frame=True)
    reference = ReferenceVideoDescription(str(root / 'lmdb'), 'ref', **options)
    optimized = LmdbEchoStudyVideoDataDescription(str(root / 'lmdb'), 'fast', **options)
    for seed in range(5):
        np.random.seed(seed)
        expected = reference.get_raw_data(ids[0])
        np.random.seed(seed)
        np.testing.assert_array_equal(optimized.get_raw_data(ids[0]), expected)
    assert optimized.start_frame == 0


@pytest.mark.parametrize('codec', ['mjpeg', 'mpeg4'])
def test_random_start_and_interframe_fallback(codec):
    avi = make_avi(45, 32, codec=codec)
    np.random.seed(123)
    reference = decode_clip(avi, 8, 3, randomize_start_frame=True, mode='sequential')
    np.random.seed(123)
    optimized = decode_clip(avi, 8, 3, randomize_start_frame=True)
    np.testing.assert_array_equal(optimized, reference)


def test_production_opencv_mjpeg_encoding(tmp_path):
    cv2 = pytest.importorskip('cv2')
    path = tmp_path / 'opencv.avi'
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'MJPG'), 30, (224, 224))
    assert writer.isOpened()
    rng = np.random.default_rng(42)
    try:
        for _ in range(37):
            gray = rng.integers(0, 256, (224, 224), dtype=np.uint8)
            writer.write(cv2.merge([gray, gray, gray]))
    finally:
        writer.release()
    avi = path.read_bytes()
    expected = decode_clip(avi, 32, 4, 5, mode='sequential', decode_threads=0)
    np.testing.assert_array_equal(decode_clip(avi, 32, 4, 5), expected)


def test_mjpeg_decodes_only_selected_frames(monkeypatch):
    avi = make_avi(60, 32)
    count = []

    class CountingPacket:
        def __init__(self, packet):
            self.packet = packet

        def __getattr__(self, name):
            return getattr(self.packet, name)

        def decode(self):
            count.append(self.packet)
            return self.packet.decode()

    class CountingContainer:
        def __init__(self, *args, **kwargs):
            self.container = av.open(*args, **kwargs)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.container.close()

        def __getattr__(self, name):
            return getattr(self.container, name)

        def demux(self, stream):
            return (CountingPacket(packet) for packet in self.container.demux(stream))

    # Extension types cannot be monkeypatched directly. Count actual packet
    # decoding through a wrapper without replacing FFmpeg or its metadata.
    from types import SimpleNamespace
    import data_descriptions.video_io as video_io
    monkeypatch.setattr(video_io, 'av', SimpleNamespace(
        open=CountingContainer,
    ))
    decode_clip(avi, 8, 4, 9)
    assert len(count) == 8


def test_missing_and_corrupt_records_close_environments(data):
    root, ids = data
    folder = root / 'lmdb'
    with pytest.raises(KeyError, match='missing'):
        read_video_bytes(folder, '100000_200000_absent')
    # Opening immediately again is invalid if a prior environment leaked.
    assert read_video_bytes(folder, ids[0])
    with pytest.raises(av.error.FFmpegError):
        decode_clip(b'not a video', 4)
    with pytest.raises(lmdb.Error):
        read_video_bytes(folder, '100000_999999_absent')
    assert not (folder / '999999.lmdb').exists()


def test_same_study_concurrent_reads_and_no_fd_growth(data):
    root, ids = data
    loader = LmdbEchoStudyVideoDataDescription(str(root / 'lmdb'), 'test', nframes=8, skip_modulo=3)
    expected = loader.get_raw_data(ids[0])
    process = psutil.Process()
    before = process.num_fds()
    with ThreadPoolExecutor(max_workers=8) as pool:
        for actual in pool.map(loader.get_raw_data, [ids[0]] * 80):
            np.testing.assert_array_equal(actual, expected)
    gc.collect()
    assert process.num_fds() <= before + 2


def build_dataset(data, multi=False, shuffle=False, transforms=None):
    root, ids = data
    loader = LmdbEchoStudyVideoDataDescription(str(root / 'lmdb'), 'test',
                                             transforms=transforms, nframes=8, skip_modulo=3)
    df = pd.read_parquet(root / 'wide.pq')
    # Use a unique target to verify video/label alignment even after shuffling.
    df['synthetic_target'] = np.arange(len(df))
    output = EcholabDataDescription(df, 'sample_id', ['synthetic_target'], 'targets',
        cls_categories_map={'synthetic_class': {0: 0, 1: 1}, 'cls_output_order': ['synthetic_class']}
        if multi else None,
        survival_task_configs=[dict(event_column='synthetic_event',
            follow_up_days_column='synthetic_follow_up_days', intervals=3,
            days_window=365, blanking_days=0)] if multi else None)
    specs = [tf.TensorSpec((2, 1), tf.float32)]
    if multi:
        specs.extend([tf.TensorSpec((2, 2), tf.float32), tf.TensorSpec((2, 6), tf.float32)])
    signature = (tf.TensorSpec((2, 8, 32, 32, 3), tf.float32), tuple(specs) if multi else specs[0])
    return make_dataset(loader, output, ids, 2, signature, shuffle, workers=4, seed=42), loader, ids


@pytest.mark.parametrize('multi', [False, True])
def test_finite_validation_and_label_alignment(data, multi):
    dataset, loader, ids = build_dataset(data, multi)
    assert int(dataset.cardinality()) == len(ids) // 2
    for _ in range(3):
        seen = []
        for videos, targets in dataset.as_numpy_iterator():
            labels = targets[0] if multi else targets
            assert labels.shape == (2, 1)  # single regression must keep its head dimension
            if multi:
                assert targets[1].shape == (2, 2)
                assert targets[2].shape == (2, 6)
            for video, label in zip(videos, labels):
                index = int(label[0])
                np.testing.assert_array_equal(video, loader.get_raw_data(ids[index]))
                seen.append(index)
        assert seen == list(range(8))  # drop_remainder agrees with recipe step counts


def test_inference_dataset_keeps_order_and_partial_batch(data):
    root, ids = data
    loader = LmdbEchoStudyVideoDataDescription(str(root / 'lmdb'), 'test', nframes=8, skip_modulo=2)
    dataset = make_inference_dataset(loader, ids, 4, (8, 32, 32, 3), workers=4)
    batches = list(dataset.as_numpy_iterator())
    assert [len(b) for b in batches] == [4, 4, 1]
    for video, sample_id in zip(np.concatenate(batches), ids):
        np.testing.assert_array_equal(video, loader.get_raw_data(sample_id))


def test_shuffle_repeat_and_cpu_augmentation(data):
    transform = RandomFlip(horizontal_prob=1, vertical_prob=0, p=1)
    dataset, loader, ids = build_dataset(data, shuffle=True, transforms=[transform])
    epochs = []
    for _ in range(2):
        order = []
        for videos, labels in dataset.as_numpy_iterator():
            for video, label in zip(videos, labels):
                index = int(label[0])
                np.testing.assert_array_equal(video, loader.get_raw_data(ids[index]))
                order.append(index)
        assert len(set(order)) == 8
        epochs.append(order)
    assert epochs[0] != epochs[1]
    assert len(list(dataset.repeat().take(9))) == 9


def test_keras_training_and_finite_validation(data):
    train, _, _ = build_dataset(data)
    valid, _, _ = build_dataset(data)
    model = tf.keras.Sequential([
        tf.keras.layers.Input((8, 32, 32, 3)),
        tf.keras.layers.GlobalAveragePooling3D(), tf.keras.layers.Dense(1),
    ])
    model.compile(optimizer='adam', loss='mse')
    history = model.fit(train.repeat().prefetch(1), validation_data=valid.prefetch(1),
                        steps_per_epoch=4, validation_steps=4, epochs=3, verbose=0)
    assert len(history.history['val_loss']) == 3
    assert np.isfinite(history.history['val_loss']).all()
