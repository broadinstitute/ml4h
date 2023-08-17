from typing import Dict

import cv2
import h5py
import logging
import numpy as np
from scipy import ndimage

from ml4h.TensorMap import TensorMap
from ml4h.normalizer import ZeroMeanStd1
from ml4h.tensormap.general import get_tensor_at_last_date, pad_or_crop_array_to_shape


def find_border_components(labeled_image):
    border_components = set()

    # Check top and bottom rows
    border_components.update(np.unique(labeled_image[0, :]))
    border_components.update(np.unique(labeled_image[-1, :]))

    # Check left and right columns
    border_components.update(np.unique(labeled_image[:, 0]))
    border_components.update(np.unique(labeled_image[:, -1]))

    # Remove background (label 0)
    border_components.discard(0)

    return border_components


def erase_border_components(image, erase_value=252):
    binary_image = (image == erase_value).astype(int)

    # Find connected components and label them
    labeled_image, num_features = ndimage.label(binary_image)

    # Find connected components adjacent to the image border
    border_components = find_border_components(labeled_image)

    for label in border_components:
        image[labeled_image == label] = 0

    return image


def dxa_background_erase(tm, hd5, dependents={}):
    tensor = get_tensor_at_last_date(hd5, tm.path_prefix, tm.name)
    bc = np.bincount(tensor.flatten().astype(np.int64))
    for val, count in zip(reversed(bc.argsort()), reversed(sorted(bc))):
        if val == 252 and count > 20000:
            tensor = erase_border_components(tensor, erase_value=252)
            break
    tensor = pad_or_crop_array_to_shape(tm.shape, tensor)
    return tensor


dxa_2 = TensorMap(
    'dxa_1_2',
    shape=(768, 768, 1),
    path_prefix='ukb_dxa',
    tensor_from_file=dxa_background_erase,
    normalization=ZeroMeanStd1(),
)
dxa_5 = TensorMap(
    'dxa_1_5',
    shape=(768, 768, 1),
    path_prefix='ukb_dxa',
    tensor_from_file=dxa_background_erase,
    normalization=ZeroMeanStd1(),
)
dxa_6 = TensorMap(
    'dxa_1_6',
    shape=(864, 768, 1),
    path_prefix='ukb_dxa',
    tensor_from_file=dxa_background_erase,
    normalization=ZeroMeanStd1(),
)
dxa_7 = TensorMap(
    'dxa_1_7',
    shape=(1792, 896, 1),
    path_prefix='ukb_dxa',
    tensor_from_file=dxa_background_erase,
    normalization=ZeroMeanStd1(),
)
dxa_8 = TensorMap(
    'dxa_1_8',
    shape=(640, 768, 1),
    path_prefix='ukb_dxa',
    tensor_from_file=dxa_background_erase,
    normalization=ZeroMeanStd1(),
)
dxa_12 = TensorMap(
    'dxa_1_12',
    shape=(928, 352, 1),
    path_prefix='ukb_dxa',
    tensor_from_file=dxa_background_erase,
    normalization=ZeroMeanStd1(),
)
dxa_11 = TensorMap(
    'dxa_1_11',
    shape=(896, 352, 1),
    path_prefix='ukb_dxa',
    tensor_from_file=dxa_background_erase,
    normalization=ZeroMeanStd1(),
)


def register_to_sample(
    register_hd5, register_path, register_name, register_shape,
    number_of_iterations = 5000, termination_eps = 1e-4, warp_mode = cv2.MOTION_TRANSLATION,
):
    register_tensor = None
    # Define termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations, termination_eps)
    with h5py.File(register_hd5, 'r') as r_hd5:
        r_tensor = get_tensor_at_last_date(r_hd5, register_path, register_name)
        register_tensor = pad_or_crop_array_to_shape(register_shape, r_tensor)

    def registered_tensor(tm, hd5, dependents={}):
        tensor = get_tensor_at_last_date(hd5, tm.path_prefix, tm.name)
        tensor = pad_or_crop_array_to_shape(tm.shape, tensor)

        bc = np.bincount(tensor.flatten().astype(np.int64))
        for val, count in zip(reversed(bc.argsort()), reversed(sorted(bc))):
            if count > 10000:
                tensor[tensor == float(val)] = 0
            else:
                break

        if warp_mode == cv2.MOTION_HOMOGRAPHY:
            warp_matrix = np.eye(3, 3, dtype=np.float32)
        else:
            warp_matrix = np.eye(2, 3, dtype=np.float32)

        try:
            (cc, warp_matrix) = cv2.findTransformECC(
                register_tensor.astype(np.float32), tensor.astype(np.float32),
                warp_matrix, warp_mode, criteria,
            )
            if warp_mode == cv2.MOTION_HOMOGRAPHY:
                # Use warpPerspective for Homography
                tensor[..., 0] = cv2.warpPerspective(
                    tensor, warp_matrix, (register_shape[1], register_shape[0]),
                    flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
                )
            else:
                # Use warpAffine for Translation, Euclidean and Affine
                tensor[..., 0] = cv2.warpAffine(
                    tensor, warp_matrix, (register_shape[1], register_shape[0]),
                    flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
                );
        except cv2.error as e:
            logging.debug(f'Got cv2 error {e}')
        return tensor
    return registered_tensor


dxa_6_translate = TensorMap(
    'dxa_1_6',
    shape=(864, 736, 1),
    path_prefix='ukb_dxa',
    tensor_from_file=register_to_sample(
        register_hd5='/mnt/disks/dxa-tensors-50k/2023_06_12/1000107.hd5',
        register_path='ukb_dxa',
        register_name='dxa_1_6',
        register_shape=(864, 736, 1),
    ),
    normalization=ZeroMeanStd1(),
)

dxa_12_translate = TensorMap(
    'dxa_1_12',
    shape=(896, 320, 1),
    path_prefix='ukb_dxa',
    tensor_from_file=register_to_sample(
        register_hd5='/mnt/disks/dxa-tensors-50k/2023_06_12/1105369.hd5',
        register_path='ukb_dxa',
        register_name='dxa_1_12',
        register_shape=(896, 320, 1),
    ),
    normalization=ZeroMeanStd1(),
)


dxa_2_homography = TensorMap(
    'dxa_1_2',
    shape=(768, 768, 1),
    path_prefix='ukb_dxa',
    tensor_from_file=register_to_sample(
        register_hd5='/mnt/disks/dxa-tensors-50k/2023_06_12/1105369.hd5',
        register_path='ukb_dxa',
        register_name='dxa_1_2',
        register_shape=(768, 768, 1),
        number_of_iterations=5000,
        termination_eps=2e-4,
        warp_mode=cv2.MOTION_HOMOGRAPHY,
    ),
    normalization=ZeroMeanStd1(),
)


dxa_5_homography = TensorMap(
    'dxa_1_5',
    shape=(768, 768, 1),
    path_prefix='ukb_dxa',
    tensor_from_file=register_to_sample(
        register_hd5='/mnt/disks/dxa-tensors-50k/2023_06_12/1105369.hd5',
        register_path='ukb_dxa',
        register_name='dxa_1_5',
        register_shape=(768, 768, 1),
        number_of_iterations=5000,
        termination_eps=2e-4,
        warp_mode=cv2.MOTION_HOMOGRAPHY,
    ),
    normalization=ZeroMeanStd1(),
)

dxa_8_homography = TensorMap(
    'dxa_1_8',
    shape=(640, 768, 1),
    path_prefix='ukb_dxa',
    tensor_from_file=register_to_sample(
        register_hd5='/mnt/disks/dxa-tensors-50k/2023_06_12/1105369.hd5',
        register_path='ukb_dxa',
        register_name='dxa_1_8',
        register_shape=(640, 768, 1),
        number_of_iterations=5000,
        termination_eps=2e-4,
        warp_mode=cv2.MOTION_HOMOGRAPHY,
    ),
    normalization=ZeroMeanStd1(),
)

dxa_12_homography = TensorMap(
    'dxa_1_12',
    shape=(928, 352, 1),
    path_prefix='ukb_dxa',
    tensor_from_file=register_to_sample(
        register_hd5='/mnt/disks/dxa-tensors-50k/2023_06_12/1105369.hd5',
        register_path='ukb_dxa',
        register_name='dxa_1_12',
        register_shape=(928, 352, 1),
        number_of_iterations=2000,
        termination_eps=2e-4,
        warp_mode=cv2.MOTION_HOMOGRAPHY,
    ),
    normalization=ZeroMeanStd1(),
)


def image_from_hd5(tm: TensorMap, hd5: h5py.File, dependents: Dict = {}) -> np.ndarray:
    return np.array(hd5[tm.name][:tm.shape[0], :tm.shape[1], :tm.shape[2]], dtype=np.float32)


dxa_1_12_registered = TensorMap('dxa_1_12_registered', shape=(928, 352, 1), tensor_from_file=image_from_hd5)
dxa_1_12_flow = TensorMap('dxa_1_12_flow', shape=(928, 352, 2), tensor_from_file=image_from_hd5)
