import argparse
import glob
import io
import logging
import os
import tarfile

import av
import cv2
import lmdb
import numpy as np
import pandas as pd
import pydicom
import skimage.measure
import skimage.morphology


def get_largest_connected_area(segmentation):
    labels = skimage.measure.label(segmentation)
    assert (labels.max() != 0)  # assume at least 1 CC
    largestCC = labels == np.argmax(np.bincount(labels.flat)[1:]) + 1
    return largestCC


def remove_small_structures(mask):
    # mask = skimage.morphology.closing(mask, np.ones((20, 20), np.float32))
    mask = skimage.morphology.opening(mask, np.ones((20, 20), np.float32))

    return mask


def get_most_central_connected_area(segmentation):
    height, width = segmentation.shape
    segmentation[height // 2:height // 2 + 200, width // 2 - 100:width // 2 + 100] = 1
    labels = skimage.measure.label(segmentation)

    assert (labels.max() != 0)  # assume at least 1 CC

    central_region = labels == np.argmax(
        np.bincount(labels[height // 2 - 100:height // 2 + 100, width // 2 - 100:width // 2 + 100].flat),
    )
    return central_region


def anonymize_echo_cone(x):
    if len(x.shape) != 4:
        raise ValueError('Only views with several frames can be anonymized')

    frames, height, width, channels = x.shape
    x_r = x[..., 0]
    temp = np.sum(x_r, 0).clip(0, 255).astype(np.uint8)
    # cone = skimage.morphology.closing(temp>100.0, np.ones((50, 50)))
    cone = get_most_central_connected_area(temp > 100.0)
    cone = remove_small_structures(cone)

    max_rows = cone.max(axis=1)
    max_cols = cone.max(axis=0)
    tallest_pixel = np.argwhere(max_rows).min()
    shortest_pixel = np.argwhere(max_rows).max()
    leftmost_pixel = np.argwhere(max_cols).min()
    rightmost_pixel = np.argwhere(max_cols).max()
    shortest_rightmost_pixel = np.argwhere(cone[:, rightmost_pixel]).max()
    tallest_pixel = min(60, tallest_pixel)

    cone[tallest_pixel:shortest_rightmost_pixel, width // 2:] = 1
    output = np.zeros_like(x_r)
    for i in range(x_r.shape[0]):
        output[i, :, :] = x_r[i, ...] * cone
    return output, leftmost_pixel


def lmdb_to_gif(lmdb_folder, view, output_path=None):
    env = lmdb.open(lmdb_folder, readonly=True, create=False)
    with env.begin(buffers=True) as txn:
        in_mem_bytes_io = io.BytesIO(txn.get(view.encode('utf-8')))
        video_container = av.open(in_mem_bytes_io, metadata_errors="ignore")
    layers = []
    for frame in video_container.decode(video=0):
        layers.append(frame.to_image())
    layers[0].save(
        f'{lmdb_folder}/{view}.gif',
        save_all=True, append_images=layers[1:], loop=0,
    )


def anonymize_echo(x):
    if (len(x.shape) != 4) or (x.shape[0] < 2):
        raise ValueError('Only views with several frames can be anonymized')

    # Some hardcoded hyperparams here, we might want to set as arguments
    blur_size = 60
    unblur_size = 40
    eps = 10
    frames, height, width, channels = x.shape
    temp = np.where(x < 5, 0, x)
    temp = np.sum(temp, 0).clip(0, 255).astype(np.uint8)

    gray = cv2.cvtColor(temp, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((blur_size, blur_size), np.float32) / (blur_size ** 2)
    filtered_gray = cv2.filter2D(gray, -1, kernel)
    ret, thresh = cv2.threshold(filtered_gray, 250, 255, cv2.THRESH_BINARY_INV)

    mask = 1 - thresh.clip(0, 1)
    mask[0:height // 10, :] = 0
    kernel = np.ones((unblur_size, unblur_size), np.float32)
    filtered_mask = cv2.filter2D(mask, -1, kernel).clip(0, 1)
    filtered_mask = np.where(filtered_mask == 0, 0, 1)
    inside_mask = np.where(filtered_mask == 1)

    left_bottom_x = min(inside_mask[1])
    right_bottom_x = max(inside_mask[1])
    top_y = min(inside_mask[0])
    left_top_x = min(inside_mask[1][inside_mask[0] == top_y])
    right_top_x = max(inside_mask[1][inside_mask[0] == top_y])
    delta = blur_size
    left_bottom_x += delta
    left_top_x -= delta
    right_top_x += delta
    right_bottom_x -= delta
    left_bottom_y = min(inside_mask[0][inside_mask[1] == left_bottom_x])
    left_top_y = min(inside_mask[0][inside_mask[1] == left_top_x])
    right_bottom_y = min(inside_mask[0][inside_mask[1] == right_bottom_x])
    right_top_y = min(inside_mask[0][inside_mask[1] == right_top_x])

    left_slope = (left_top_y - left_bottom_y) / (left_top_x - left_bottom_x)
    left_x_intercept = -left_bottom_y / left_slope + left_bottom_x
    leftmost = [left_slope, left_x_intercept]
    right_slope = (right_top_y - right_bottom_y) / (right_top_x - right_bottom_x)
    right_x_intercept = -right_bottom_y / right_slope + right_bottom_x
    rightmost = [right_slope, right_x_intercept]

    m1, m2 = np.meshgrid(np.arange(width), np.arange(height))
    # use epsilon to avoid masking part of the echo
    mask = leftmost[0] * (m1 - leftmost[1]) - eps < m2
    mask *= rightmost[0] * (m1 - rightmost[1]) - eps < m2
    mask = np.reshape(mask, (height, width)).astype(np.int8)
    mask[top_y + delta:] = 0
    filtered_mask += mask
    filtered_mask = filtered_mask.clip(0, 1)

    max_rows = filtered_mask.max(axis=1)
    max_cols = filtered_mask.max(axis=0)
    tallest_pixel = np.argwhere(max_rows).min()
    leftmost_pixel = np.argwhere(max_cols).min()
    tallest_pixel = max(80, tallest_pixel)
    filtered_mask[tallest_pixel:, width // 2:] = 1

    output = np.zeros_like(x)
    for i in range(frames):
        for c in range(channels):
            output[i, :, :, c] = x[i, :, :, c] * filtered_mask
    return output, leftmost_pixel


def array_to_cropped_avi(array, output_path, fps, target_size, leftmost_pixel=0):
    frames, height, width = array.shape

    if frames < 2:
        raise ValueError('You cannot save a video with no frames')

    fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    out = cv2.VideoWriter(output_path, fourcc, fps, target_size)

    for i in range(frames):
        outputA = array[i, :, :]
        max_width = max(width, leftmost_pixel + height)
        smallOutput = outputA[:, leftmost_pixel:max_width]

        # Resize image
        output = cv2.resize(smallOutput, target_size, interpolation=cv2.INTER_CUBIC)

        finaloutput = cv2.merge([output, output, output])
        out.write(finaloutput)
    out.release()


def targz_to_avis(study_id, study_folder, tar):
    if os.path.isfile(os.path.join(study_folder, f'{study_id}.lmdb/data.mdb')):
        logging.warning(f'Skipping {study_id} as it already exists')
        return

    log_dic = {'study': [], 'view': [], 'log': [], 'stored': []}
    target_size = (224, 224)

    if tar:
        try:
            with tarfile.open(os.path.join(study_folder, f'{study_id}.tar.gz'), 'r:gz') as targz:
                targz.extractall(study_folder)
        except:
            logging.warning(f'Extraction failed for {study_folder}{study_id}.tar.gz')
            return

    dicom_paths = glob.glob(os.path.join(study_folder, str(study_id), '*'))
    env = lmdb.open(os.path.join(study_folder, f'{study_id}.lmdb'), map_size=2 ** 32 - 1)
    with env.begin(write=True) as txn:
        for dicom_path in dicom_paths:
            dicom_filename = os.path.basename(dicom_path)
            log_dic['view'].append(dicom_filename)
            log_dic['study'].append(study_id)
            logging.info(f'Reading {dicom_path}')
            try:
                dcm = pydicom.dcmread(dicom_path, force=True)
                testarray = dcm.pixel_array
                testarray_anon, leftmost_pixel = anonymize_echo_cone(testarray)
            except Exception as e:
                error_msg = f'{dicom_path}: {e}'
                log_dic['log'].append(error_msg)
                log_dic['stored'].append(False)
                logging.warning(error_msg)
                os.remove(dicom_path)
                continue

            fps = 30
            try:
                fps = dcm['CineRate'].value
            except:
                logging.info("Could not find frame rate, default to 30")

            video_path = os.path.join(study_folder, f'{study_id}.lmdb', f'{dicom_filename}.avi')
            array_to_cropped_avi(
                testarray_anon,
                video_path,
                fps,
                target_size,
                leftmost_pixel,
            )

            # Save avi into lmdb
            with open(video_path, 'rb') as avi:
                logging.info(f'Adding {dicom_filename} to the transaction')
                txn.put(key=dicom_filename.encode('utf-8'), value=avi.read())
                log_dic['log'].append('')
                log_dic['stored'].append(True)
                logging.info(f"Successfully stored {dicom_filename} into LMDB")

    log_df = pd.DataFrame(log_dic)
    log_df.to_parquet(os.path.join(study_folder, f'{study_id}.lmdb', f'log_{study_id}.pq'))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--df_path', help='path to a csv that must contain a column named study with unique identifiers'
        'that correspond to directory names in the study_folder',
    )
    parser.add_argument('--study_folder', help='path to the directory containing one subdirectory per echo study')
    parser.add_argument('--start', default=-1, help='optional, row in csv of the study to start processing')
    parser.add_argument('--end', default=-1, help='optional, row in csv of the study to end processing (inclusive)')
    parser.add_argument('--tar', action='store_true', help='indicates that study folders are stored as .tar.gz files')
    return parser.parse_args()


def main(**kwargs):
    df_path = kwargs['df_path']
    study_folder = kwargs['study_folder']
    start = int(kwargs.get('start'))
    end = int(kwargs.get('end'))
    tar = kwargs.get('tar')

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    df = pd.read_csv(df_path)
    for i, row in df.iterrows():
        if i < start:
            continue
        if -1 < end <= i:
            continue
        study = int(row['study'])
        targz_to_avis(study, study_folder, tar)


if __name__ == '__main__':
    ARGS = parse_args()
    main(**vars(ARGS))
