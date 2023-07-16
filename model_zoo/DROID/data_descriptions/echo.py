import os
import io
import av
import itertools

import lmdb

import numpy as np
import pandas as pd

from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

from ml4ht.data.data_description import DataDescription

VIEW_OPTION_KEY = 'view'

metadata_elements = ['PhotometricInterpretation',
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
            start_beat=0,
    ):

        self.local_lmdb_dir = local_lmdb_dir
        self._name = name
        self.nframes = nframes
        self.nframes = (nframes + start_beat) * skip_modulo
        self.start_beat = start_beat
        # transformations
        self.transforms = transforms or []
        self.skip_modulo = skip_modulo

    def get_loading_options(self, sample_id):
        _, study, view = sample_id.split('_')
        lmdb_folder = os.path.join(self.local_lmdb_dir, f"{study}.lmdb")
        lmdb_log = pd.read_parquet(os.path.join(lmdb_folder, f'log_{study}.pq')).set_index('view')
        lmdb_log = lmdb_log[lmdb_log['stored']]

        if view not in lmdb_log.index:
            raise ValueError('View not saved in the LMDB')

        return [
            {VIEW_OPTION_KEY: view}
        ]

    def get_raw_data(self, sample_id, loading_option=None):
        try:
            sample_id = sample_id.decode('UTF-8')
        except (UnicodeDecodeError, AttributeError):
            pass
        _, study, view = sample_id.split('_')

        lmdb_folder = os.path.join(self.local_lmdb_dir, f"{study}.lmdb")

        env = lmdb.open(lmdb_folder, readonly=True, lock=False)
        nframes = self.nframes

        frames = []
        with env.begin(buffers=True) as txn:
            in_mem_bytes_io = io.BytesIO(txn.get(view.encode('utf-8')))
            video_container = av.open(in_mem_bytes_io, metadata_errors="ignore")
            video_frames = itertools.cycle(video_container.decode(video=0))
            for i, frame in enumerate(video_frames):
                if i == nframes:
                    break
                if i < (self.start_beat * self.skip_modulo):
                    continue
                if self.skip_modulo > 1:
                    if (i % self.skip_modulo) != 0:
                        continue
                frame = np.array(frame.to_image())
                for transform in self.transforms:
                    frame = transform(frame, loading_option)
                frames.append(frame)
            del video_frames
            video_container.close()
        env.close()
        return np.squeeze(np.array(frames, dtype='float32') / 255.)

    @property
    def name(self):
        return self._name
