import os
import io
import av
import itertools

import lmdb

import numpy as np
import pandas as pd

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
