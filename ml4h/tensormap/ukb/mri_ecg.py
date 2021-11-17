import logging
from typing import Dict, Tuple

import numpy as np
from tensorflow.keras.utils import to_categorical

from ml4h.TensorMap import TensorMap, Interpretation
from ml4h.defines import LAX_4CH_HEART_LABELS, LAX_4CH_MYOCARDIUM_LABELS, MRI_LAX_4CH_SEGMENTED_CHANNEL_MAP, ECG_REST_MEDIAN_LEADS
from ml4h.normalizer import ZeroMeanStd1, Standardize
from ml4h.tensormap.general import get_tensor_at_first_date, pad_or_crop_array_to_shape


def _make_ecg_rest(
        hd5, path_prefix: str, ecg_shape: Tuple[int], ecg_leads: Dict[str, int], instance: int = 2, downsample_steps: int = 0,
):
    tensor = np.zeros(ecg_shape, dtype=np.float32)
    for k in hd5[path_prefix]:
        if k in ecg_leads:
            data = hd5[f'{path_prefix}/{k}/instance_{instance}']
            if downsample_steps > 1:
                tensor[:, ecg_leads[k]] = np.array(data, dtype=np.float32)[::downsample_steps]
            tensor[:, ecg_leads[k]] = pad_or_crop_array_to_shape((ecg_shape[0],), data)
    return tensor


def _heart_mask_and_ecg_instances(mri_path_prefix, mri_shape, mri_key, mri_segmentation_key, mri_labels, mri_normalizer,
                                  ecg_prefix, ecg_shape, ecg_leads, ecg_normalizer, include_mri_segmentation=False, total_instances=50):
    def _heart_mask_tensor_from_file(tm, hd5, dependents={}):
        diastole_categorical = get_tensor_at_first_date(hd5, mri_path_prefix, f'{mri_segmentation_key}{1}')
        heart_mask = np.isin(diastole_categorical, list(mri_labels.values()))
        i, j = np.where(heart_mask)
        indices = np.meshgrid(np.arange(min(i), max(i) + 1), np.arange(min(j), max(j) + 1), np.arange(total_instances), indexing='ij')
        segmentation_indices = np.meshgrid(np.arange(min(i), max(i) + 1), np.arange(min(j), max(j) + 1), indexing='ij')
        mri = mri_normalizer.normalize(get_tensor_at_first_date(hd5, mri_path_prefix, f'{mri_key}'))
        if ecg_shape[0] > 0:
            ecg = ecg_normalizer.normalize(_make_ecg_rest(hd5, ecg_prefix, ecg_shape, ecg_leads))
        tensor = np.zeros(tm.shape, dtype=np.float32)
        for frame in range(1, total_instances+1):
            if include_mri_segmentation:
                frame_categorical = get_tensor_at_first_date(hd5, mri_path_prefix, f'{mri_segmentation_key}{frame}')
                heart_mask = np.isin(frame_categorical, list(mri_labels.values()))
                mri[:mri.shape[0], :mri.shape[1], frame-1] = heart_mask[:mri.shape[0], :mri.shape[1]] * mri[..., frame-1]
                frame_categorical = get_tensor_at_first_date(hd5, mri_path_prefix, f'{mri_segmentation_key}{frame}')
                reshape_categorical = pad_or_crop_array_to_shape(mri_shape[:2], frame_categorical[segmentation_indices])
                if len(mri_shape) == 4:
                    slice_one_hot = to_categorical(reshape_categorical, tm.shape[-1]-1)
                    tensor[:mri_shape[0], :mri_shape[1], frame - 1, 1:] = slice_one_hot
                else:
                    tensor[mri_shape[0]:, :mri_shape[1], frame - 1] = reshape_categorical
            if ecg_shape[0] > 0:
                ecg_start = (frame-1) * (ecg_shape[0] // total_instances)
                ecg_stop = frame * (ecg_shape[0] // total_instances)
                for lead in ecg_leads:
                    lead_index = ecg_leads[lead] + mri_shape[0]
                    if len(mri_shape) == 3:
                        tensor[lead_index, :, frame-1] = np.repeat(ecg[ecg_start:ecg_stop, ecg_leads[lead]], tm.shape[1]//(ecg_stop-ecg_start))
                    elif len(mri_shape) == 4:
                        tensor[lead_index, :, frame - 1, :] = np.expand_dims(np.repeat(ecg[ecg_start:ecg_stop, ecg_leads[lead]], tm.shape[1] // (ecg_stop - ecg_start)), axis=-1)
        if len(mri_shape) == 3:
            tensor[:mri_shape[0], :mri_shape[1], :mri_shape[2]] = pad_or_crop_array_to_shape(mri_shape, mri[tuple(indices)])
        elif len(mri_shape) == 4:
            tensor[:mri_shape[0], :mri_shape[1], :mri_shape[2], 0] = pad_or_crop_array_to_shape(mri_shape[:3], mri[tuple(indices)])
        return tensor
    return _heart_mask_tensor_from_file


tff = _heart_mask_and_ecg_instances('ukb_cardiac_mri', (96, 96, 50), 'cine_segmented_lax_4ch/2/',
                                    'cine_segmented_lax_4ch_annotated_', LAX_4CH_HEART_LABELS, ZeroMeanStd1(),
                                    'ukb_ecg_rest', (600, 12), ECG_REST_MEDIAN_LEADS, Standardize(mean=0, std=10))
ecg_and_lax_4ch = TensorMap(
    'ecg_and_lax_4ch', Interpretation.CONTINUOUS, shape=(108, 96, 50),
    tensor_from_file=tff,
)
tff = _heart_mask_and_ecg_instances('ukb_cardiac_mri', (92, 96, 50), 'cine_segmented_lax_4ch/2/',
                                    'cine_segmented_lax_4ch_annotated_', LAX_4CH_HEART_LABELS, ZeroMeanStd1(),
                                    'ukb_ecg_rest', (600, 12), ECG_REST_MEDIAN_LEADS, Standardize(mean=0, std=10))
ecg_and_lax_4ch_104 = TensorMap(
    'ecg_and_lax_4ch_104', Interpretation.CONTINUOUS, shape=(104, 96, 50),
    tensor_from_file=tff,
)

tff = _heart_mask_and_ecg_instances('ukb_cardiac_mri', (96, 96, 50, len(MRI_LAX_4CH_SEGMENTED_CHANNEL_MAP)+1), 'cine_segmented_lax_4ch/2/',
                                    'cine_segmented_lax_4ch_annotated_', LAX_4CH_HEART_LABELS, ZeroMeanStd1(),
                                    'ukb_ecg_rest', (600, 12), ECG_REST_MEDIAN_LEADS, Standardize(mean=0, std=10),
                                    include_mri_segmentation=True)
ecg_seg_lax_4ch = TensorMap(
    'ecg_seg_lax_4ch', Interpretation.CONTINUOUS, shape=(108, 96, 50, len(MRI_LAX_4CH_SEGMENTED_CHANNEL_MAP)+1),
    tensor_from_file=tff, channel_map=MRI_LAX_4CH_SEGMENTED_CHANNEL_MAP,
)

tff = _heart_mask_and_ecg_instances('ukb_cardiac_mri', (64, 64, 50, len(MRI_LAX_4CH_SEGMENTED_CHANNEL_MAP)+1), 'cine_segmented_lax_4ch/2/',
                                    'cine_segmented_lax_4ch_annotated_', LAX_4CH_HEART_LABELS, ZeroMeanStd1(),
                                    'ukb_ecg_rest', (0, 0), ECG_REST_MEDIAN_LEADS, Standardize(mean=0, std=10),
                                    include_mri_segmentation=True)
seg_lax_4ch_64xy_50z_18c = TensorMap(
    'seg_lax_4ch_64xy_50z_18c', Interpretation.CONTINUOUS, shape=(64, 64, 50, len(MRI_LAX_4CH_SEGMENTED_CHANNEL_MAP)+1),
    tensor_from_file=tff, channel_map=MRI_LAX_4CH_SEGMENTED_CHANNEL_MAP,
)

tff = _heart_mask_and_ecg_instances('ukb_cardiac_mri', (64, 64, 50), 'cine_segmented_lax_4ch/2/',
                                    'cine_segmented_lax_4ch_annotated_', LAX_4CH_HEART_LABELS, ZeroMeanStd1(),
                                    'ukb_ecg_rest', (0, 0), ECG_REST_MEDIAN_LEADS, Standardize(mean=0, std=10),
                                    include_mri_segmentation=True)
seg_lax_4ch_128x_64y_50z_1c = TensorMap(
    'seg_lax_4ch_128x_64y_50z_1c', Interpretation.CONTINUOUS, shape=(128, 64, 50),
    tensor_from_file=tff, channel_map=MRI_LAX_4CH_SEGMENTED_CHANNEL_MAP,
)