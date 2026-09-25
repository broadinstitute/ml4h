"""Bounded-lifetime LMDB reads and selective decoding for DROID's MJPEG AVIs."""

import io
import itertools
import os
import threading

import av
import lmdb
import numpy as np


# python-lmdb forbids simultaneously opening the same environment twice in a
# process. Serialize only short read/copy/close sections for a given study;
# decode runs outside the lock. Stripes bound lock memory regardless of studies.
_READ_LOCKS = [threading.Lock() for _ in range(64)]


def read_video_bytes(local_lmdb_dir, sample_id):
    if isinstance(sample_id, bytes):
        sample_id = sample_id.decode('utf-8')
    _, study, view = str(sample_id).split('_')
    path = os.path.realpath(os.path.join(local_lmdb_dir, f'{study}.lmdb'))
    with _READ_LOCKS[hash(path) % len(_READ_LOCKS)]:
        with lmdb.open(path, readonly=True, lock=False, create=False) as env:
            with env.begin() as txn:
                value = txn.get(view.encode('utf-8'))
                if value is None:
                    raise KeyError(f'Video key {view!r} is missing from {path}')
                return value  # bytes own their memory after the environment closes


def decode_clip(video_bytes, nframes, skip_modulo=1, start_frame=0,
                randomize_start_frame=False, mode='auto', decode_threads=1):
    """Return float32 THWC in [0, 1], preserving cyclic short-clip padding.

    AVI MJPEG has independent frames. Validate one indexed key packet per frame
    before skipping compressed packets. Other codecs/containers use sequential
    decoding, since inter-frame codecs require their reference frames.
    ``sequential`` also provides an A/B path for the pre-change algorithm.
    """
    if nframes is None or nframes <= 0 or skip_modulo <= 0 or start_frame < 0:
        raise ValueError('nframes/skip_modulo must be positive; start_frame must be non-negative')
    if mode not in ('auto', 'sequential') or decode_threads < 0:
        raise ValueError('Invalid decode mode or thread count')
    with av.open(io.BytesIO(video_bytes), metadata_errors='ignore') as container:
        stream = container.streams.video[0]
        stream.thread_type = 'AUTO'
        stream.thread_count = decode_threads
        packets = None
        if mode == 'auto' and stream.codec_context.name == 'mjpeg' and container.format.name == 'avi':
            candidates = [p for p in container.demux(stream) if p.size]
            if (stream.frames > 0 and len(candidates) == stream.frames
                    and all(p.is_keyframe for p in candidates)):
                packets = candidates
            else:
                container.seek(0)

        if randomize_start_frame:
            total = stream.frames
            if not total:
                total = sum(1 for _ in container.decode(video=0))
                container.seek(0)
            frame_range = total - nframes * skip_modulo
            if frame_range > 0:
                # Deliberately retain the existing exclusive upper bound.
                start_frame = np.random.randint(frame_range)

        if packets is not None:
            # Frame threading delays output; selective packet decode needs
            # immediate output. Keep the stream's codec metadata/extradata.
            stream.thread_count = 1
            indices = (start_frame + np.arange(nframes) * skip_modulo) % len(packets)
            frames = {}
            for index in np.unique(indices):
                decoded = packets[int(index)].decode()
                if len(decoded) != 1:
                    # Unusual packetization: let the generic decoder establish
                    # frame order. Do not resample a random start on fallback.
                    return decode_clip(video_bytes, nframes, skip_modulo, start_frame,
                                       mode='sequential', decode_threads=decode_threads)
                frames[index] = decoded[0].to_ndarray(format='rgb24')
            video = np.stack([frames[index] for index in indices]).astype(np.float32)
        else:
            frames = []
            cycling = itertools.cycle(container.decode(video=0))
            for index, frame in enumerate(cycling):
                if index >= start_frame and (index - start_frame) % skip_modulo == 0:
                    frames.append(frame.to_ndarray(format='rgb24'))
                    if len(frames) == nframes:
                        break
            del cycling
            if not frames:
                raise ValueError('Video contains no decodable frames')
            video = np.asarray(frames, dtype=np.float32)
        video /= 255.0
        return video
