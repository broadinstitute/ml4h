import h5py
import nibabel
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
t1_mni_slice_80_216 = TensorMap(
    'axial_80',
    shape=(216, 216, 1),
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
t1_slice_100_224 = TensorMap(
    'axial_100',
    shape=(224, 256, 1),
    path_prefix='ukb_brain_mri/T1/',
    tensor_from_file=normalized_first_date,
    normalization=ZeroMeanStd1(),
)
t1_slice_100_256 = TensorMap(
    'axial_100',
    shape=(256, 256, 1),
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

def make_brain_volume_tensor_fxn(steps = 1):
    def _brain_volume_from_file(tm, hd5, dependents={}):
        tensor = np.zeros(tm.shape, dtype=np.float32)
        begin_slice = int(tm.name.split('_')[-2])
        end_slice = int(tm.name.split('_')[-1])
        for i in range(begin_slice, end_slice, steps):
            slicer = get_tensor_at_first_date(hd5, tm.path_prefix, f'axial_{i}')
            tensor[..., (i-begin_slice)//steps] = pad_or_crop_array_to_shape((tm.shape[0], tm.shape[1]), slicer)
        return tensor
    return _brain_volume_from_file


t1_slices_68_100 = TensorMap(
    'axial_68_100',
    shape=(216, 256, 32),
    path_prefix='ukb_brain_mri/T1/',
    tensor_from_file=make_brain_volume_tensor_fxn(),
    normalization=ZeroMeanStd1(),
)

t1_slices_68_100 = TensorMap(
    'axial_68_100',
    shape=(216, 256, 32),
    path_prefix='ukb_brain_mri/T1/',
    tensor_from_file=make_brain_volume_tensor_fxn(),
    normalization=ZeroMeanStd1(),
)
t1_slices_32_64 = TensorMap(
    't1_axial_32_64',
    shape=(216, 256, 32),
    path_prefix='ukb_brain_mri/T1/',
    tensor_from_file=make_brain_volume_tensor_fxn(),
    normalization=ZeroMeanStd1(),
)
t1_slices_64_96 = TensorMap(
    't1_axial_64_96',
    shape=(216, 256, 32),
    path_prefix='ukb_brain_mri/T1/',
    tensor_from_file=make_brain_volume_tensor_fxn(),
    normalization=ZeroMeanStd1(),
)
t1_slices_96_128 = TensorMap(
    't1_axial_96_128',
    shape=(216, 256, 32),
    path_prefix='ukb_brain_mri/T1/',
    tensor_from_file=make_brain_volume_tensor_fxn(),
    normalization=ZeroMeanStd1(),
)
t1_slices_96_99 = TensorMap(
    't1_axial_96_99',
    shape=(216, 256, 3),
    path_prefix='ukb_brain_mri/T1/',
    tensor_from_file=make_brain_volume_tensor_fxn(),
    normalization=ZeroMeanStd1(),
)

t1_mni_slices_96_99 = TensorMap(
    'axial_96_99',
    shape=(176, 216, 3),
    path_prefix='ukb_brain_mri/T1_brain_to_MNI/',
    tensor_from_file=make_brain_volume_tensor_fxn(),
    normalization=ZeroMeanStd1(),
)
t1_mni_slices_16_48 = TensorMap(
    'axial_16_48',
    shape=(176, 216, 32),
    path_prefix='ukb_brain_mri/T1_brain_to_MNI/',
    tensor_from_file=make_brain_volume_tensor_fxn(),
    normalization=ZeroMeanStd1(),
)
t1_mni_slices_48_80 = TensorMap(
    'axial_48_80',
    shape=(176, 216, 32),
    path_prefix='ukb_brain_mri/T1_brain_to_MNI/',
    tensor_from_file=make_brain_volume_tensor_fxn(),
    normalization=ZeroMeanStd1(),
)
t1_mni_slices_80_144 = TensorMap(
    'axial_80_144',
    shape=(176, 216, 64),
    path_prefix='ukb_brain_mri/T1_brain_to_MNI/',
    tensor_from_file=make_brain_volume_tensor_fxn(),
    normalization=ZeroMeanStd1(),
)
t1_mni_slices_0_182 = TensorMap(
    'axial_0_182',
    shape=(176, 216, 182),
    path_prefix='ukb_brain_mri/T1_brain_to_MNI/',
    tensor_from_file=make_brain_volume_tensor_fxn(),
    normalization=ZeroMeanStd1(),
)

t1_mni_slices_32_160_step4 = TensorMap(
    'axial_32_160',
    shape=(176, 216, 32),
    path_prefix='ukb_brain_mri/T1_brain_to_MNI/',
    tensor_from_file=make_brain_volume_tensor_fxn(steps=4),
    normalization=ZeroMeanStd1(),
)

t1_mni_slices_0_32 = TensorMap(
    'axial_0_32',
    shape=(176, 216, 32),
    path_prefix='ukb_brain_mri/T1_brain_to_MNI/',
    tensor_from_file=make_brain_volume_tensor_fxn(),
    normalization=ZeroMeanStd1(),
)

t1_mni_slices_32_64 = TensorMap(
    'axial_32_64',
    shape=(176, 216, 32),
    path_prefix='ukb_brain_mri/T1_brain_to_MNI/',
    tensor_from_file=make_brain_volume_tensor_fxn(),
    normalization=ZeroMeanStd1(),
)
t1_mni_slices_64_96 = TensorMap(
    'axial_64_96',
    shape=(176, 216, 32),
    path_prefix='ukb_brain_mri/T1_brain_to_MNI/',
    tensor_from_file=make_brain_volume_tensor_fxn(),
    normalization=ZeroMeanStd1(),
)
t1_mni_slices_96_128 = TensorMap(
    'axial_96_128',
    shape=(176, 216, 32),
    path_prefix='ukb_brain_mri/T1_brain_to_MNI/',
    tensor_from_file=make_brain_volume_tensor_fxn(),
    normalization=ZeroMeanStd1(),
)
t1_mni_slices_128_160 = TensorMap(
    'axial_128_160',
    shape=(176, 216, 32),
    path_prefix='ukb_brain_mri/T1_brain_to_MNI/',
    tensor_from_file=make_brain_volume_tensor_fxn(),
    normalization=ZeroMeanStd1(),
)

t1_mni_slices_160_192 = TensorMap(
    'axial_160_192',
    shape=(176, 216, 32),
    path_prefix='ukb_brain_mri/T1_brain_to_MNI/',
    tensor_from_file=make_brain_volume_tensor_fxn(),
    normalization=ZeroMeanStd1(),
)
t1_mni_slices_30_60 = TensorMap(
    'axial_30_60',
    shape=(176, 216, 30),
    path_prefix='ukb_brain_mri/T1_brain_to_MNI/',
    tensor_from_file=make_brain_volume_tensor_fxn(),
    normalization=ZeroMeanStd1(),
)
t1_mni_slices_10_73 = TensorMap(
    'axial_10_73',
    shape=(176, 216, 63),
    path_prefix='ukb_brain_mri/T1_brain_to_MNI/',
    tensor_from_file=make_brain_volume_tensor_fxn(),
    normalization=ZeroMeanStd1(),
)
t2_flair_orig_defaced_slices_0_32 = TensorMap(
    't2_flair_axial_0_32',
    shape=(192, 256, 32),
    path_prefix='ukb_brain_mri/T2_FLAIR_orig_defaced/',
    tensor_from_file=make_brain_volume_tensor_fxn(),
    normalization=ZeroMeanStd1(),
)
t2_flair_orig_defaced_slices_32_64 = TensorMap(
    't2_flair_axial_32_64',
    shape=(192, 256, 32),
    path_prefix='ukb_brain_mri/T2_FLAIR_orig_defaced/',
    tensor_from_file=make_brain_volume_tensor_fxn(),
    normalization=ZeroMeanStd1(),
)
t2_flair_orig_defaced_slices_64_96 = TensorMap(
    't2_flair_axial_64_96',
    shape=(192, 256, 32),
    path_prefix='ukb_brain_mri/T2_FLAIR_orig_defaced/',
    tensor_from_file=make_brain_volume_tensor_fxn(),
    normalization=ZeroMeanStd1(),
)
t2_flair_orig_defaced_slices_96_128 = TensorMap(
    't2_flair_axial_96_128',
    shape=(192, 256, 32),
    path_prefix='ukb_brain_mri/T2_FLAIR_orig_defaced/',
    tensor_from_file=make_brain_volume_tensor_fxn(),
    normalization=ZeroMeanStd1(),
)
t2_flair_orig_defaced_slices_128_160 = TensorMap(
    't2_flair_axial_128_160',
    shape=(192, 256, 32),
    path_prefix='ukb_brain_mri/T2_FLAIR_orig_defaced/',
    tensor_from_file=make_brain_volume_tensor_fxn(),
    normalization=ZeroMeanStd1(),
)
t2_flair_orig_defaced_slices_160_192 = TensorMap(
    't2_flair_axial_160_192',
    shape=(192, 256, 32),
    path_prefix='ukb_brain_mri/T2_FLAIR_orig_defaced/',
    tensor_from_file=make_brain_volume_tensor_fxn(),
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


brain_channel_map = {
    'Background': 0, 'Left_Thalamus_Proper': 1, 'Left_Caudate': 2, 'Left_Putamen': 3, 'Left_Pallidum': 4, 'Brain_Stem': 5, 'Left_Hippocampus': 6,
    'Left_Amygdala': 7, 'Left_Accumbens_area': 8, 'Right_Thalamus_Proper': 9, 'Right_Caudate': 10, 'Right_Putamen': 11, 'Right_Pallidum': 12,
    'Right_Hippocampus': 13, 'Right_Amygdala': 14, 'Right_Accumbens_area': 15,
}

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


def _mni_label_masked(labels, mni_label_mask='/home/sam/mni_icbm152_CerebrA_tal_nlin_sym_09c.nii'):
    mni_nifti = nibabel.load('/home/sam/mni_icbm152_CerebrA_tal_nlin_sym_09c.nii')
    mni_labels = mni_nifti.get_fdata()[:176, :216, :182]  # crop to UKB MNI

    def _masked_brain_tensor(tm, hd5, dependents={}):
        tensor = np.zeros(tm.shape, dtype=np.float32)
        begin_slice = int(tm.name.split('_')[-2])
        end_slice = int(tm.name.split('_')[-1])
        for i in range(begin_slice, end_slice):
            slicer = get_tensor_at_first_date(hd5, tm.path_prefix, f'axial_{i}')
            tensor[..., i - begin_slice] = pad_or_crop_array_to_shape((tm.shape[0], tm.shape[1]), slicer)
        label_mask = np.isin(mni_labels[..., begin_slice:end_slice], list(labels.values()))
        tensor *= label_mask
        return tensor
    return _masked_brain_tensor


t1_mni_slice_80_hippocampus = TensorMap(
    'mni_hippocampus_axial_48_80',
    shape=(176, 216, 32),
    path_prefix='ukb_brain_mri/T1_brain_to_MNI/',
    tensor_from_file=_mni_label_masked({'Left_Hippocampus': 99, 'Right_Hippocampus': 48}),  # CerebrA Label Map
    normalization=ZeroMeanStd1(),
)
t1_mni_slice_60_92_putamen = TensorMap(
    'mni_putamen_axial_60_92',
    shape=(176, 216, 32),
    path_prefix='ukb_brain_mri/T1_brain_to_MNI/',
    tensor_from_file=_mni_label_masked({'Left_Putamen': 72, 'Right_Putamen': 21}),  # CerebrA Label Map
    normalization=ZeroMeanStd1(),
)
t1_mni_cerebellum_white_matter_30_60 = TensorMap(
    't1_mni_cerebellum_white_matter_30_60',
    shape=(176, 216, 30),
    path_prefix='ukb_brain_mri/T1_brain_to_MNI/',
    tensor_from_file=_mni_label_masked({'Left_cerebellum_white_matter': 90, 'Right_cerebellum_white_matter': 39}),
    normalization=ZeroMeanStd1(),
)
t1_mni_cerebellum_gray_matter_30_60 = TensorMap(
    't1_mni_cerebellum_gray_matter_30_60',
    shape=(176, 216, 30),
    path_prefix='ukb_brain_mri/T1_brain_to_MNI/',
    tensor_from_file=_mni_label_masked({'Left_cerebellum_gray_matter': 97, 'Right_cerebellum_gray_matter': 46}),
    normalization=ZeroMeanStd1(),
)
t1_mni_cerebellum_30_60 = TensorMap(
    't1_mni_cerebellum_30_60',
    shape=(176, 216, 30),
    path_prefix='ukb_brain_mri/T1_brain_to_MNI/',
    tensor_from_file=_mni_label_masked({
        'Left_cerebellum_gray_matter': 97, 'Right_cerebellum_gray_matter': 46,
        'Left_cerebellum_white_matter': 90, 'Right_cerebellum_white_matter': 39,
    }),
    normalization=ZeroMeanStd1(),
)
t1_mni_cerebellum_gray_matter_10_73 = TensorMap(
    't1_mni_cerebellum_gray_matter_10_73',
    shape=(176, 216, 63),
    path_prefix='ukb_brain_mri/T1_brain_to_MNI/',
    tensor_from_file=_mni_label_masked({'Left_cerebellum_gray_matter': 97, 'Right_cerebellum_gray_matter': 46}),
    normalization=ZeroMeanStd1(),
)
t1_mni_cerebellum_gray_matter_10_73 = TensorMap(
    't1_mni_cerebellum_gray_matter_10_73',
    shape=(176, 216, 63),
    path_prefix='ukb_brain_mri/T1_brain_to_MNI/',
    tensor_from_file=_mni_label_masked({'Left_cerebellum_gray_matter': 97, 'Right_cerebellum_gray_matter': 46}),
    normalization=ZeroMeanStd1(),
)
t1_mni_cerebellum_10_73 = TensorMap(
    't1_mni_cerebellum_10_73',
    shape=(176, 216, 63),
    path_prefix='ukb_brain_mri/T1_brain_to_MNI/',
    tensor_from_file=_mni_label_masked({
        'Left_cerebellum_gray_matter': 97, 'Right_cerebellum_gray_matter': 46,
        'Left_cerebellum_white_matter': 90, 'Right_cerebellum_white_matter': 39,
    }),
    normalization=ZeroMeanStd1(),
)
t1_mni_amygdala_47_71 = TensorMap(
    't1_mni_amygdala_47_71',
    shape=(176, 216, 24),
    path_prefix='ukb_brain_mri/T1_brain_to_MNI/',
    tensor_from_file=_mni_label_masked({'Left_amygdala': 19, 'Right_amygdala': 70}),
    normalization=ZeroMeanStd1(),
)
t1_mni_accumbens_65_76 = TensorMap(
    't1_mni_accumbens_65_76',
    shape=(176, 216, 11),
    path_prefix='ukb_brain_mri/T1_brain_to_MNI/',
    tensor_from_file=_mni_label_masked({'Left_accumbens': 19, 'Right_accumbens': 70}),
    normalization=ZeroMeanStd1(),
)
def _random_slice_bounded(low=0, high=192):
    def random_mni_slice_tensor(tm, hd5, dependents={}):
        slice_index = np.random.randint(low, high)
        tensor = pad_or_crop_array_to_shape(
                tm.shape, np.array(
                tm.hd5_first_dataset_in_group(hd5, f'{tm.path_prefix}axial_{slice_index}/'), dtype=np.float32,
            ),
        )
        dependents[tm.dependent_map] = np.zeros(
            tm.dependent_map.shape,
            dtype=np.float32,
        )
        dependents[tm.dependent_map][0] = (float(slice_index) - 96) / 14.0
        return tensor
    return random_mni_slice_tensor

axial_index_map = TensorMap('axial_index', Interpretation.CONTINUOUS, shape=(1,), channel_map={'axial_index':0})

t1_mni_random_slice = TensorMap(
    't1_mni_random_slice',
    Interpretation.CONTINUOUS,
    shape=(192, 192, 1),
    path_prefix='ukb_brain_mri/T1_brain_to_MNI/',
    tensor_from_file=_random_slice_bounded(),
    normalization=ZeroMeanStd1(),
    dependent_map=axial_index_map,
)

t1_mni_random_slice_64_128 = TensorMap(
    't1_mni_random_slice',
    Interpretation.CONTINUOUS,
    shape=(192, 192, 1),
    path_prefix='ukb_brain_mri/T1_brain_to_MNI/',
    tensor_from_file=_random_slice_bounded(64, 128),
    normalization=ZeroMeanStd1(),
    dependent_map=axial_index_map,
)

t1_mni_random_slice_48_144 = TensorMap(
    't1_mni_random_slice',
    Interpretation.CONTINUOUS,
    shape=(192, 192, 1),
    path_prefix='ukb_brain_mri/T1_brain_to_MNI/',
    tensor_from_file=_random_slice_bounded(48, 144),
    normalization=ZeroMeanStd1(),
    dependent_map=axial_index_map,
)

t1_mni_random_slice_32_160 = TensorMap(
    't1_mni_random_slice',
    Interpretation.CONTINUOUS,
    shape=(192, 192, 1),
    path_prefix='ukb_brain_mri/T1_brain_to_MNI/',
    tensor_from_file=_random_slice_bounded(32, 160),
    normalization=ZeroMeanStd1(),
    dependent_map=axial_index_map,
)

t1_random_slice = TensorMap(
    't1_random_slice',
    Interpretation.CONTINUOUS,
    shape=(224, 256, 1),
    path_prefix='ukb_brain_mri/T1/',
    tensor_from_file=_random_slice_bounded(6, 200),
    normalization=ZeroMeanStd1(),
    dependent_map=axial_index_map,
)

t1_random_slice_256 = TensorMap(
    't1_random_slice_256',
    Interpretation.CONTINUOUS,
    shape=(256, 256, 1),
    path_prefix='ukb_brain_mri/T1/',
    tensor_from_file=_random_slice_bounded(16, 192),
    normalization=ZeroMeanStd1(),
    dependent_map=axial_index_map,
)

t2_flair_random_slice = TensorMap(
    't2_flair_random_slice',
    Interpretation.CONTINUOUS,
    shape=(192, 256, 1),
    path_prefix='ukb_brain_mri/T2_FLAIR_orig_defaced/',
    tensor_from_file=_random_slice_bounded(64, 128),
    normalization=ZeroMeanStd1(),
    dependent_map=axial_index_map,
)

t2_flair_random_slice_32_192 = TensorMap(
    't2_flair_random_slice',
    Interpretation.CONTINUOUS,
    shape=(192, 256, 1),
    path_prefix='ukb_brain_mri/T2_FLAIR_orig_defaced/',
    tensor_from_file=_random_slice_bounded(32, 192),
    normalization=ZeroMeanStd1(),
    dependent_map=axial_index_map,
)

t2_flair_random_slice_127 = TensorMap(
    't2_flair_random_slice',
    Interpretation.CONTINUOUS,
    shape=(192, 256, 1),
    path_prefix='ukb_brain_mri/T2_FLAIR_orig_defaced/',
    tensor_from_file=_random_slice_bounded(127, 128),
    normalization=ZeroMeanStd1(),
    dependent_map=axial_index_map,
)
