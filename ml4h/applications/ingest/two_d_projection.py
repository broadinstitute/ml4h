import time
import json
from multiprocessing import Pool, cpu_count
from typing import Dict, List, Tuple

import os
import h5py
import numpy as np
import pandas as pd
from scipy.ndimage import zoom
from fastparquet import ParquetFile
import matplotlib.pyplot as plt

from ingest_mri import compress_and_store, read_compressed


def project_coronal(x):
    return x.mean(axis=0).T


def project_sagittal(x):
    return x.mean(axis=1).T


def normalize(projection):
    projection = 255 * projection / np.sort(projection)[:-50].mean()
    return projection.astype(np.uint16)


def stack(xs):
    return np.vstack(xs)


def center_pad(x, width: int):
    """pad an image on the left and right with 0s to a target width"""
    new_x = np.zeros((x.shape[0], width))
    offset = (width - x.shape[1]) // 2
    new_x[:, offset: width - offset] = x
    return new_x


def center_pad_stack(xs):
    """center and pad then stack images with different widths"""
    max_width = max(x.shape[1] for x in xs)
    return np.vstack([center_pad(x, max_width) for x in xs])


def build_z_slices(
        num_slices: List[int],
        z_pos: List[Tuple[float, float]],
) -> List[slice]:
    """
    Finds which images to take from each station.
    Takes a few noisy images from the top of each station
    and removes any remaining overlap between stations.

    num_slices: number of slices of each station
    z_pos: min z, max z of each station

    Example calculating the top and bottom z position of each station:
        z_pos = [50, 100], [90, 140]
        -> z_pos = [54, 100], [94, 140]  # remove images from the top
        -> overlap = [94, 100]
        -> z_pos = [54, 100], [100, 140]  # remove overlapping images
    """
    # remove a few slices from the top of each station
    top_remove_num = 4
    slice_starts = []
    for i in range(len(num_slices)):
        slice_starts.append(top_remove_num)
        top_remove_frac = top_remove_num / num_slices[i]
        lo, hi = z_pos[i]
        z_pos[i] = lo, hi - (hi - lo) * top_remove_frac

    # remove remaining overlaps from the bottom of each station
    slice_ends = []
    for i in range(0, len(num_slices) - 1):
        series_below_min_z, series_below_max_z = z_pos[i + 1]
        series_above_min_z, series_above_max_z = z_pos[i]
        overlap_size = series_below_max_z - series_above_min_z
        overlap_frac = overlap_size / (series_above_max_z - series_above_min_z)
        slice_ends.append(int((1 - overlap_frac) * num_slices[i]))
    slice_ends.append(num_slices[-1])  # the last station gets nothing removed from the bottom
    # build slices
    return [slice(start, end) for start, end in zip(slice_starts, slice_ends)]


def build_projections(data: Dict[int, np.ndarray], meta_data: pd.DataFrame) -> Dict[str, np.ndarray]:
    """
    Build coronal projections for each series type from all of the series
    :param data: {series number: series array}
    :param meta_data: meta data loaded from parquet file
    :return: {series type: coronal projection}
    """
    # stations are differently scaled on the z-axis
    station_z_scales = 3.0, 4.5, 4.5, 4.5, 3.5, 4.0
    station_z_scales = [scale / 3 for scale in station_z_scales]

    z_pos = meta_data.groupby('series_number')['image_position_z'].agg(['min', 'max'])
    slices = build_z_slices(
        [data[i].shape[-1] for i in range(1, 25, 4)],
        [z_pos.loc[i] for i in range(1, 25, 4)],
    )

    # keep track of where stations are connected
    horizontal_lines = [(idx.stop - idx.start) * scale for idx, scale in zip(slices, station_z_scales)]
    horizontal_lines = np.cumsum(horizontal_lines).astype(np.uint16)[:-1]
    projections = {'horizontal_line_idx': horizontal_lines}

    # build coronal and sagittal projections
    for type_idx, series_type_name in zip(range(4), ('in', 'opp', 'f', 'w')):
        coronal_to_stack = []
        sagittal_to_stack = []
        for station_idx in range(1, 25, 4):  # neck, upper ab, lower ab, legs
            series_num = station_idx + type_idx
            station_slice = slices[station_idx // 4]
            scale = station_z_scales[station_idx // 4]
            coronal = project_coronal(data[series_num][..., station_slice])
            coronal = zoom(coronal, (scale, 1.), order=1)  # account for z axis scaling
            coronal_to_stack.append(coronal)
            sagittal = project_sagittal(data[series_num][..., station_slice])
            sagittal = zoom(sagittal, (scale, 1.), order=1)  # account for z axis scaling
            sagittal_to_stack.append(sagittal)

        projections[f'{series_type_name}_coronal'] = normalize(stack(coronal_to_stack))
        projections[f'{series_type_name}_sagittal'] = normalize(center_pad_stack(sagittal_to_stack))
    return projections


def visualize_projections(
    projections: Dict[str, np.ndarray],
    output_path: str,
):
    h, w = projections['in_coronal'].shape
    w *= 4
    h *= 2
    fig, (axes1, axes2) = plt.subplots(2, 4, figsize=(w // 40, h // 40))
    cor = sorted(name for name in projections if 'cor' in name)
    sag = sorted(name for name in projections if 'sag' in name)
    horiz = projections['horizontal_line_idx']
    for ax, name in zip(axes1, cor):
        ax.set_title(name)
        ax.set_axis_off()
        ax.imshow(projections[name], cmap='gray')
        for height in horiz:
            ax.axhline(height, c='r', linestyle='--')
    for ax, name in zip(axes2, sag):
        ax.set_title(name)
        ax.set_axis_off()
        ax.imshow(projections[name], cmap='gray')
        for height in horiz:
            ax.axhline(height, c='r', linestyle='--')
    fig.savefig(output_path, bbox_inches='tight', pad_inches=0)


def build_projection_hd5(
        old_hd5_path: str,
        output_folder: str,
):
    new_path = os.path.join(output_folder, os.path.basename(old_hd5_path))
    with h5py.File(old_hd5_path, 'r') as old_hd5:
        for instance in old_hd5['instance']:
            data = {
                int(name): read_compressed(old_hd5[f'instance/{instance}/series/{name}'])
                for name in old_hd5[f'instance/{instance}/series']
            }
            meta_path = old_hd5_path.replace('.h5', f'_{instance}.pq')
            meta = ParquetFile(meta_path).to_pandas()
            projection = build_projections(data, meta)
            with h5py.File(new_path, 'a') as new_hd5:
                for name, im in projection.items():
                    compress_and_store(new_hd5, im, f'instance/{instance}/{name}')


def _build_projection_hd5s(hd5_files: List[str], destination: str):
    errors = {}
    name = os.getpid()
    print(f'Starting process {name} with {len(hd5_files)} files')
    for i, path in enumerate(hd5_files):
        try:
            build_projection_hd5(path, destination)
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
