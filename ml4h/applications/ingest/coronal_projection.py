import time
import json
from multiprocessing import Pool, cpu_count
from typing import Dict, List

import os
import h5py
import numpy as np
import pandas as pd
from fastparquet import ParquetFile
import matplotlib.pyplot as plt

from ingest_mri import compress_and_store, read_compressed


def project(x):
    return x.mean(axis=0).T


def stack(xs):
    return np.vstack(xs)


def _num_images_to_drop(
        series_above_max_z: float, series_above_min_z: float, series_above_num_slices: int,
        series_below_max_z: float,
) -> int:
    """
    Calculates the number of images to drop from the series above
    so that the series align in the final image
    """
    overlap_size = series_below_max_z - series_above_min_z
    overlap_frac = overlap_size / (series_above_max_z - series_above_min_z)
    return int(max(0., overlap_frac) * series_above_num_slices)


def build_projections(data: Dict[int, np.ndarray], meta_data: pd.DataFrame) -> Dict[str, np.ndarray]:
    """
    Build coronal projections for each series type from all of the series
    :param data: {series number: series array}
    :param meta_data: meta data loaded from parquet file
    :return: {series type: coronal projection}
    """
    z_pos = meta_data.groupby('series_number')['image_position_z'].agg(['min', 'max'])
    projections = {}
    for type_idx, series_type_name in zip(range(4), ('in', 'opp', 'f', 'w')):
        to_stack = []
        for station_idx in range(1, 25, 4):  # neck, upper ab, lower ab, legs
            series_num = station_idx + type_idx
            if station_idx == 21:  # don't drop anything from the last station
                to_stack.append(project(data[series_num]))
                continue
            z_lo, z_hi = z_pos.loc[series_num]
            next_z_lo, next_z_hi = z_pos.loc[series_num + 4]
            drop_num = _num_images_to_drop(
                series_above_max_z=z_hi, series_above_min_z=z_lo,
                series_above_num_slices=data[series_num].shape[-1],
                series_below_max_z=next_z_hi,
            )
            if drop_num <= 0:  # in this case don't drop anything
                drop_num = data[series_num].shape[-1]
            to_stack.append(project(data[series_num][..., :-drop_num]))
        projections[series_type_name] = stack(to_stack).astype(np.uint16)
    return projections


def visualize_projections(
    projections: Dict[str, np.ndarray],
    output_path: str,
):
    h, w = projections['in'].shape
    w *= 4
    fig, axes = plt.subplots(1, 4, figsize=(w // 40, h // 40))
    for ax, (name, im) in zip(axes, projections.items()):
        ax.set_title(name)
        ax.set_axis_off()
        ax.imshow(im, cmap='gray')
    fig.savefig(output_path, bbox_inches='tight', pad_inches=0)


def build_projection_hd5(
        old_hd5_path: str,
        old_parquet_path: str,
        output_folder: str,
):
    new_path = os.path.join(output_folder, os.path.basename(old_hd5_path))
    meta = ParquetFile(old_parquet_path).to_pandas()
    with h5py.File(old_hd5_path, 'r') as old_hd5, h5py.File(new_path, 'w') as new_hd5:
        if len(old_hd5['instance']) != 1:
            raise ValueError('Meta data was not stored correctly for multi-instance data.')
        for instance in old_hd5['instance']:
            data = {
                int(name): read_compressed(old_hd5[f'instance/{instance}/series/{name}'])
                for name in old_hd5[f'instance/{instance}/series']
            }
            projection = build_projections(data, meta)
            for name, im in projection.items():
                compress_and_store(new_hd5, im, f'instance/{instance}/{name}')


def _build_projection_hd5s(hd5_files: List[str], destination: str):
    errors = {}
    name = os.getpid()
    print(f'Starting process {name} with {len(hd5_files)} files')
    for i, path in enumerate(hd5_files):
        pq_path = path.replace('.h5', '.pq')
        try:
            build_projection_hd5(path, pq_path, destination)
        except Exception as e:
            errors[path] = str(e)
        if len(hd5_files) % max(i // 10, 1) == 0:
            print(f'{name}: {(i + 1) / len(hd5_files):.2%} done')
    return errors


def multiprocess_project(
    hd5_files: List[str],
    destination: str,
):
    os.makedirs(destination, exist_ok=True)
    split_files = np.array_split(hd5_files, cpu_count())
    print(f'Beginning coronal projection of {len(hd5_files)} samples.')
    start = time.time()
    errors = {}
    with Pool(cpu_count()) as pool:
        results = [pool.apply_async(_build_projection_hd5s, (split, destination)) for split in split_files]
        for result in results:
            errors.update(result.get())
    delta = time.time() - start
    print(f'Projections took {delta:.1f} seconds at {delta / len(hd5_files):.1f} s/file')
    with open(os.path.join(destination, 'errors.json'), 'w') as f:
        json.dump(errors, f)
    return errors
