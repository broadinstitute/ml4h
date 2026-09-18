"""Generate synthetic MJPEG AVIs in the same per-study LMDB layout as DROID."""

import argparse
import io
import json
from pathlib import Path

import av
import lmdb
import numpy as np
import pandas as pd


def make_avi(nframes=160, size=224, seed=0, codec='mjpeg'):
    """Moving, textured grayscale sector; all pixels are generated, never clinical."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:size, :size].astype(np.float32)
    x, y = (xx - size / 2) / size, yy / size
    sector = (np.abs(x) < y * 0.65) & (y > 0.08) & (x*x + y*y < 0.95)
    texture = rng.uniform(0, 90, (size, size))
    buffer = io.BytesIO()
    with av.open(buffer, mode='w', format='avi') as container:
        stream = container.add_stream(codec, rate=30)
        stream.width = stream.height = size
        stream.pix_fmt = 'yuvj420p' if codec == 'mjpeg' else 'yuv420p'
        stream.thread_count = 1
        for t in range(nframes):
            phase = 2 * np.pi * t / 30
            radius = np.sqrt((x - 0.08 * np.sin(phase))**2 + (y - 0.5)**2)
            ring = 110 * np.exp(-((radius - 0.17 - 0.02 * np.cos(phase)) / 0.025)**2)
            noise = rng.uniform(0, 25, (size, size))
            gray = ((texture + ring + noise + 15) * sector).clip(0, 255).astype(np.uint8)
            rgb = np.repeat(gray[..., None], 3, axis=-1)
            for packet in stream.encode(av.VideoFrame.from_ndarray(rgb, format='rgb24')):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return buffer.getvalue()


def generate(root, studies=24, views=4, frames=160, size=224, seed=0):
    root = Path(root)
    # Exclusive directory creation prevents accidental overwrites of any dataset.
    root.mkdir(parents=True, exist_ok=False)
    lmdb_root = root / 'lmdb'
    lmdb_root.mkdir()
    rows = []
    for study_idx in range(studies):
        patient, study = 100000 + study_idx, 200000 + study_idx
        folder = lmdb_root / f'{study}.lmdb'
        log = []
        with lmdb.open(str(folder), map_size=2**30) as env:
            for view_idx in range(views):
                view = f'clip{view_idx:03d}.dcm'
                avi = make_avi(frames, size, seed + study_idx * views + view_idx)
                with env.begin(write=True) as txn:
                    txn.put(view.encode(), avi)
                log.append(dict(study=study, view=view, stored=True, log='synthetic'))
                rows.append(dict(
                    sample_id=f'{patient}_{study}_{view}', view_prediction=9,
                    doppler_prediction=0, quality_prediction=0, canonical_prediction=0,
                    synthetic_target=float(study_idx % 10), synthetic_class=view_idx % 2,
                    synthetic_event=study_idx % 2, synthetic_follow_up_days=100 + study_idx * 20,
                ))
        pd.DataFrame(log).to_parquet(folder / f'log_{study}.pq', index=False)
    pd.DataFrame(rows).to_parquet(root / 'wide.pq', index=False)
    split = max(1, min(studies - 1, int(studies * 0.8)))
    (root / 'splits.json').write_text(json.dumps(dict(
        patient_train=list(range(100000, 100000 + split)),
        patient_valid=list(range(100000 + split, 100000 + studies)),
    ), indent=2))
    (root / 'manifest.json').write_text(json.dumps(dict(
        synthetic=True, studies=studies, views=views, frames=frames, size=size,
        seed=seed, codec='mjpeg', container='avi', samples=len(rows),
    ), indent=2))
    return rows


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, help='New directory; never overwrites.')
    parser.add_argument('--studies', type=int, default=24)
    parser.add_argument('--views', type=int, default=4)
    parser.add_argument('--frames', type=int, default=160)
    parser.add_argument('--size', type=int, default=224)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    if min(args.studies, args.views, args.frames, args.size) <= 0:
        parser.error('All dimensions must be positive.')
    rows = generate(args.output, args.studies, args.views, args.frames, args.size, args.seed)
    print(f'Created {len(rows)} synthetic AVI records in {args.output}')
