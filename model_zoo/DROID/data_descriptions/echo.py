import os
import numpy as np
import pandas as pd

from ml4ht.data.data_description import DataDescription
from .video_io import decode_clip, read_video_bytes

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
            decode_mode='auto',
            decode_threads=1,
    ):

        self.local_lmdb_dir = local_lmdb_dir
        self._name = name
        self.start_frame = start_frame
        self.nframes = nframes
        # transformations
        self.transforms = transforms or []
        self.skip_modulo = skip_modulo
        self.randomize_start_frame = randomize_start_frame
        self.decode_mode = decode_mode
        self.decode_threads = decode_threads

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
        video = decode_clip(
            read_video_bytes(self.local_lmdb_dir, sample_id), self.nframes,
            self.skip_modulo, self.start_frame, self.randomize_start_frame,
            self.decode_mode, self.decode_threads,
        )

        # Transforms operate on the whole normalized clip of shape (T, H, W, C) in
        # [0, 1] so that augmentations requiring cross-frame consistency (jitter,
        # rotation, flips, mask sector) can share sampled parameters across frames.
        # See data_descriptions/transforms.py for the transform contract.
        for transform in self.transforms:
            video = transform(video, loading_option)
        return np.asarray(video, dtype='float32')

    @property
    def name(self):
        return self._name
