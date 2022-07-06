import h5py
import numpy as np
from tensorflow.keras.utils import to_categorical

from ml4h.normalizer import ZeroMeanStd1
from ml4h.metrics import weighted_crossentropy
from ml4h.TensorMap import TensorMap, Interpretation
from ml4h.tensormap.general import get_tensor_at_first_date, normalized_first_date, pad_or_crop_array_to_shape


t1_slice_80 = TensorMap(
    'axial_80',
    shape=(216, 256, 1),
    path_prefix='ukb_brain_mri/T1/',
    tensor_from_file=normalized_first_date,
    normalization=ZeroMeanStd1(),
)
t1_mni_slice_48 = TensorMap(
    'axial_48',
    shape=(176, 216, 1),
    path_prefix='ukb_brain_mri/T1_brain_to_MNI/',
    tensor_from_file=normalized_first_date,
    normalization=ZeroMeanStd1(),
)

t1_mni_slice_80 = TensorMap(
    'axial_80',
    shape=(176, 216, 1),
    path_prefix='ukb_brain_mri/T1_brain_to_MNI/',
    tensor_from_file=normalized_first_date,
    normalization=ZeroMeanStd1(),
)
t1_slice_85 = TensorMap(
    'axial_85',
    shape=(216, 256, 1),
    path_prefix='ukb_brain_mri/T1/',
    tensor_from_file=normalized_first_date,
    normalization=ZeroMeanStd1(),
)

t1_slice_100 = TensorMap(
    'axial_100',
    shape=(216, 256, 1),
    path_prefix='ukb_brain_mri/T1/',
    tensor_from_file=normalized_first_date,
    normalization=ZeroMeanStd1(),
)
t1_slice_120 = TensorMap(
    'axial_120',
    shape=(216, 256, 1),
    path_prefix='ukb_brain_mri/T1/',
    tensor_from_file=normalized_first_date,
    normalization=ZeroMeanStd1(),
)
t1_slice_160 = TensorMap(
    'axial_160',
    shape=(216, 256, 1),
    path_prefix='ukb_brain_mri/T1/',
    tensor_from_file=normalized_first_date,
    normalization=ZeroMeanStd1(),
)


def _brain_volume_from_file(tm, hd5, dependents={}):
    tensor = np.zeros(tm.shape, dtype=np.float32)
    begin_slice = int(tm.name.split('_')[-2])
    end_slice = int(tm.name.split('_')[-1])
    for i in range(begin_slice, end_slice):
        slicer = get_tensor_at_first_date(hd5, tm.path_prefix, f'axial_{i}')
        tensor[..., i-begin_slice] = pad_or_crop_array_to_shape((tm.shape[0], tm.shape[1]), slicer)
    return tensor


t1_slices_68_100 = TensorMap(
    'axial_68_100',
    shape=(216, 256, 32),
    path_prefix='ukb_brain_mri/T1/',
    tensor_from_file=_brain_volume_from_file,
    normalization=ZeroMeanStd1(),
)

t1_mni_slices_48_80 = TensorMap(
    'axial_48_80',
    shape=(176, 216, 32),
    path_prefix='ukb_brain_mri/T1_brain_to_MNI/',
    tensor_from_file=_brain_volume_from_file,
    normalization=ZeroMeanStd1(),
)


def _segmented_brain_tensor_from_file(tm, hd5, dependents={}):
    # from mapping given in https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FIRST/UserGuide
    num2idx = {0: 0, 10: 1, 11: 2, 12: 3, 13: 4, 16: 5, 17: 6, 18: 7, 26: 8, 49: 9, 50: 10, 51: 11, 52: 12, 53: 13, 54: 14, 58: 15}
    tensor = np.zeros(tm.shape, dtype=np.float32)
    if tm.axes() == 3:
        categorical_index_slice = get_tensor_at_first_date(hd5, tm.path_prefix, tm.name)
        index_remap = np.zeros_like(categorical_index_slice)
        for key in num2idx:
            index_remap[categorical_index_slice == key] = num2idx[key]
        categorical_one_hot = to_categorical(index_remap, len(tm.channel_map))
        tensor[..., :] = pad_or_crop_array_to_shape(tensor[..., :].shape, categorical_one_hot)
    else:
        raise ValueError(f'No method to get segmented slices for TensorMap: {tm}')
    return tensor


brain_channel_map = {'Background': 0, 'Left_Thalamus_Proper': 1, 'Left_Caudate': 2, 'Left_Putamen': 3, 'Left_Pallidum': 4, 'Brain_Stem': 5, 'Left_Hippocampus': 6,
                     'Left_Amygdala': 7, 'Left_Accumbens_area': 8, 'Right_Thalamus_Proper': 9, 'Right_Caudate': 10, 'Right_Putamen': 11, 'Right_Pallidum': 12,
                     'Right_Hippocampus': 13, 'Right_Amygdala': 14, 'Right_Accumbens_area': 15}

t1_seg_slice_80 = TensorMap(
    'axial_80',
    interpretation=Interpretation.CATEGORICAL,
    shape=(216, 256, len(brain_channel_map)),
    path_prefix='ukb_brain_mri/T1_first_all_fast_firstseg/',
    channel_map=brain_channel_map,
    loss=weighted_crossentropy([0.1]+[10.0]*15),
    tensor_from_file=_segmented_brain_tensor_from_file,
)

t1_seg_slice_85 = TensorMap(
    'axial_85',
    interpretation=Interpretation.CATEGORICAL,
    shape=(216, 256, len(brain_channel_map)),
    path_prefix='ukb_brain_mri/T1_first_all_fast_firstseg/',
    channel_map=brain_channel_map,
    tensor_from_file=_segmented_brain_tensor_from_file,
)


def _brain_label_masked(labels, segmentation_key='ukb_brain_mri/T1_first_all_fast_firstseg/'):
    def _masked_brain_tensor(tm, hd5, dependents={}):
        # from mapping given in https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/FIRST/UserGuide
        num2idx = {0: 0, 10: 1, 11: 2, 12: 3, 13: 4, 16: 5, 17: 6, 18: 7, 26: 8, 49: 9, 50: 10, 51: 11, 52: 12, 53: 13, 54: 14, 58: 15}
        tensor = np.zeros(tm.shape, dtype=np.float32)
        if tm.axes() == 3:
            mri = get_tensor_at_first_date(hd5, tm.path_prefix, tm.name)
            categorical_index_slice = get_tensor_at_first_date(hd5, segmentation_key, tm.name)
            index_remap = np.zeros_like(categorical_index_slice)
            for key in num2idx:
                index_remap[categorical_index_slice == key] = num2idx[key]

            label_mask = np.isin(index_remap, list(labels.values()))
            mri = label_mask * mri
            tensor = pad_or_crop_array_to_shape(tensor.shape, mri)
        else:
            raise ValueError(f'No method to get segmented slices for TensorMap: {tm}')
        return tensor
    return _masked_brain_tensor


t1_slice_80_hippocampus = TensorMap(
    'axial_80',
    shape=(216, 256, 1),
    path_prefix='ukb_brain_mri/T1/',
    tensor_from_file=_brain_label_masked({'Left_Hippocampus': 6, 'Right_Hippocampus': 13}),
    normalization=ZeroMeanStd1(),
)
t1_slice_92_putamen = TensorMap(
    'axial_92',
    shape=(216, 256, 1),
    path_prefix='ukb_brain_mri/T1/',
    tensor_from_file=_brain_label_masked({'Left_Putamen': 3, 'Right_Putamen': 11}),
    normalization=ZeroMeanStd1(),
)
t1_slice_63_brainstem = TensorMap(
    'axial_63',
    shape=(216, 256, 1),
    path_prefix='ukb_brain_mri/T1/',
    tensor_from_file=_brain_label_masked({'Brain_Stem': 5}),
    normalization=ZeroMeanStd1(),
)