# MNIST Hand written digit tensor maps
import os
import sys
import gzip
import pickle
import logging
from typing import Dict

import h5py
import numpy as np

import tensorflow as tf
from skimage.transform import resize
from ml4h.TensorMap import TensorMap, Interpretation
from ml4h.normalizer import ZeroMeanStd1


def image_from_hd5(tm: TensorMap, hd5: h5py.File, dependents: Dict = {}) -> np.ndarray:
    return np.array(hd5[tm.name][:tm.shape[0], :tm.shape[1], :tm.shape[2]], dtype=np.float32)


mnist_image = TensorMap('mnist_image', shape=(28, 28, 1), tensor_from_file=image_from_hd5)
celeba_image = TensorMap('celeba_image', shape=(218, 178, 3), normalization=ZeroMeanStd1(),
                         tensor_from_file=image_from_hd5)
celeba_image_208 = TensorMap('celeba_image', shape=(208, 168, 3), normalization=ZeroMeanStd1(),
                         tensor_from_file=image_from_hd5)



def downsampled_image_from_hd5(tm: TensorMap, hd5: h5py.File, dependents: Dict = {}) -> np.ndarray:
    dependents[tm.dependent_map] = np.zeros(tm.dependent_map.shape, dtype=np.float32)
    dependents[tm.dependent_map][:] = np.array(hd5[tm.dependent_map.name][:tm.dependent_map.shape[0], :tm.dependent_map.shape[1], :tm.dependent_map.shape[2]], dtype=np.float32)
    logging.info(f'GOT dependent data {tm.dependent_map} shape: {dependents[tm.dependent_map].shape}')
    resized = resize(dependents[tm.dependent_map], (tm.shape[0], tm.shape[1]))
    logging.info(f'GOT resize data {resized.shape}')
    return resized


celeba_image_208_downsample = TensorMap('celeba_image_208_downsample', shape=(26, 21, 3), normalization=ZeroMeanStd1(),
                         tensor_from_file=downsampled_image_from_hd5, dependent_map=celeba_image_208)

def landmark_from_hd5(tm: TensorMap, hd5: h5py.File, dependents: Dict = {}) -> np.ndarray:
    landmark = np.zeros(tm.shape, dtype=np.float32)
    for channel in tm.channel_map:
        landmark[tm.channel_map[channel]] = hd5[channel][0]
    return landmark

eye_channels = {'lefteye_x': 0, 'lefteye_y': 1, 'righteye_x': 2, 'righteye_y': 3}
celeba_eyes = TensorMap('celeba_eyes', shape=(len(eye_channels),), channel_map=eye_channels,
                        tensor_from_file=landmark_from_hd5)


def celeba_label(tm: TensorMap, hd5: h5py.File, dependents: Dict = {}) -> np.ndarray:
    if tm.name in hd5:
        return np.array([1, 0])
    else:
        return np.array([0, 1])


celeba_arched_eyebrows = TensorMap('arched_eyebrows', Interpretation.CATEGORICAL, tensor_from_file=celeba_label, channel_map={'no_arched_eyebrows': 0, 'arched_eyebrows': 1})
celeba_attractive = TensorMap('attractive', Interpretation.CATEGORICAL, tensor_from_file=celeba_label, channel_map={'no_attractive': 0, 'attractive': 1})
celeba_bags_under_eyes = TensorMap('bags_under_eyes', Interpretation.CATEGORICAL, tensor_from_file=celeba_label, channel_map={'no_bags_under_eyes': 0, 'bags_under_eyes': 1})
celeba_bald = TensorMap('bald', Interpretation.CATEGORICAL, tensor_from_file=celeba_label, channel_map={'no_bald': 0, 'bald': 1})
celeba_bangs = TensorMap('bangs', Interpretation.CATEGORICAL, tensor_from_file=celeba_label, channel_map={'no_bangs': 0, 'bangs': 1})
celeba_big_lips = TensorMap('big_lips', Interpretation.CATEGORICAL, tensor_from_file=celeba_label, channel_map={'no_big_lips': 0, 'big_lips': 1})
celeba_big_nose = TensorMap('big_nose', Interpretation.CATEGORICAL, tensor_from_file=celeba_label, channel_map={'no_big_nose': 0, 'big_nose': 1})
celeba_black_hair = TensorMap('black_hair', Interpretation.CATEGORICAL, tensor_from_file=celeba_label, channel_map={'no_black_hair': 0, 'black_hair': 1})
celeba_blond_hair = TensorMap('blond_hair', Interpretation.CATEGORICAL, tensor_from_file=celeba_label, channel_map={'no_blond_hair': 0, 'blond_hair': 1})
celeba_blurry = TensorMap('blurry', Interpretation.CATEGORICAL, tensor_from_file=celeba_label, channel_map={'no_blurry': 0, 'blurry': 1})
celeba_brown_hair = TensorMap('brown_hair', Interpretation.CATEGORICAL, tensor_from_file=celeba_label, channel_map={'no_brown_hair': 0, 'brown_hair': 1})
celeba_bushy_eyebrows = TensorMap('bushy_eyebrows', Interpretation.CATEGORICAL, tensor_from_file=celeba_label, channel_map={'no_bushy_eyebrows': 0, 'bushy_eyebrows': 1})
celeba_chubby = TensorMap('chubby', Interpretation.CATEGORICAL, tensor_from_file=celeba_label, channel_map={'no_chubby': 0, 'chubby': 1})
celeba_double_chin = TensorMap('double_chin', Interpretation.CATEGORICAL, tensor_from_file=celeba_label, channel_map={'no_double_chin': 0, 'double_chin': 1})
celeba_eyeglasses = TensorMap('eyeglasses', Interpretation.CATEGORICAL, tensor_from_file=celeba_label, channel_map={'no_eyeglasses': 0, 'eyeglasses': 1})
celeba_goatee = TensorMap('goatee', Interpretation.CATEGORICAL, tensor_from_file=celeba_label, channel_map={'no_goatee': 0, 'goatee': 1})
celeba_gray_hair = TensorMap('gray_hair', Interpretation.CATEGORICAL, tensor_from_file=celeba_label, channel_map={'no_gray_hair': 0, 'gray_hair': 1})
celeba_heavy_makeup = TensorMap('heavy_makeup', Interpretation.CATEGORICAL, tensor_from_file=celeba_label, channel_map={'no_heavy_makeup': 0, 'heavy_makeup': 1})
celeba_high_cheekbones = TensorMap('high_cheekbones', Interpretation.CATEGORICAL, tensor_from_file=celeba_label, channel_map={'no_high_cheekbones': 0, 'high_cheekbones': 1})
celeba_male = TensorMap('male', Interpretation.CATEGORICAL, tensor_from_file=celeba_label, channel_map={'no_male': 0, 'male': 1})
celeba_mouth_slightly_open = TensorMap('mouth_slightly_open', Interpretation.CATEGORICAL, tensor_from_file=celeba_label, channel_map={'no_mouth_slightly_open': 0, 'mouth_slightly_open': 1})
celeba_mustache = TensorMap('mustache', Interpretation.CATEGORICAL, tensor_from_file=celeba_label, channel_map={'no_mustache': 0, 'mustache': 1})
celeba_narrow_eyes = TensorMap('narrow_eyes', Interpretation.CATEGORICAL, tensor_from_file=celeba_label, channel_map={'no_narrow_eyes': 0, 'narrow_eyes': 1})
celeba_no_beard = TensorMap('no_beard', Interpretation.CATEGORICAL, tensor_from_file=celeba_label, channel_map={'no_no_beard': 0, 'no_beard': 1})
celeba_oval_face = TensorMap('oval_face', Interpretation.CATEGORICAL, tensor_from_file=celeba_label, channel_map={'no_oval_face': 0, 'oval_face': 1})
celeba_pale_skin = TensorMap('pale_skin', Interpretation.CATEGORICAL, tensor_from_file=celeba_label, channel_map={'no_pale_skin': 0, 'pale_skin': 1})
celeba_pointy_nose = TensorMap('pointy_nose', Interpretation.CATEGORICAL, tensor_from_file=celeba_label, channel_map={'no_pointy_nose': 0, 'pointy_nose': 1})
celeba_receding_hairline = TensorMap('receding_hairline', Interpretation.CATEGORICAL, tensor_from_file=celeba_label, channel_map={'no_receding_hairline': 0, 'receding_hairline': 1})
celeba_rosy_cheeks = TensorMap('rosy_cheeks', Interpretation.CATEGORICAL, tensor_from_file=celeba_label, channel_map={'no_rosy_cheeks': 0, 'rosy_cheeks': 1})
celeba_sideburns = TensorMap('sideburns', Interpretation.CATEGORICAL, tensor_from_file=celeba_label, channel_map={'no_sideburns': 0, 'sideburns': 1})
celeba_smiling = TensorMap('smiling', Interpretation.CATEGORICAL, tensor_from_file=celeba_label, channel_map={'no_smiling': 0, 'smiling': 1})
celeba_straight_hair = TensorMap('straight_hair', Interpretation.CATEGORICAL, tensor_from_file=celeba_label, channel_map={'no_straight_hair': 0, 'straight_hair': 1})
celeba_wavy_hair = TensorMap('wavy_hair', Interpretation.CATEGORICAL, tensor_from_file=celeba_label, channel_map={'no_wavy_hair': 0, 'wavy_hair': 1})
celeba_wearing_earrings = TensorMap('wearing_earrings', Interpretation.CATEGORICAL, tensor_from_file=celeba_label, channel_map={'no_wearing_earrings': 0, 'wearing_earrings': 1})
celeba_wearing_hat = TensorMap('wearing_hat', Interpretation.CATEGORICAL, tensor_from_file=celeba_label, channel_map={'no_wearing_hat': 0, 'wearing_hat': 1})
celeba_wearing_lipstick = TensorMap('wearing_lipstick', Interpretation.CATEGORICAL, tensor_from_file=celeba_label, channel_map={'no_wearing_lipstick': 0, 'wearing_lipstick': 1})
celeba_wearing_necklace = TensorMap('wearing_necklace', Interpretation.CATEGORICAL, tensor_from_file=celeba_label, channel_map={'no_wearing_necklace': 0, 'wearing_necklace': 1})
celeba_wearing_necktie = TensorMap('wearing_necktie', Interpretation.CATEGORICAL, tensor_from_file=celeba_label, channel_map={'no_wearing_necktie': 0, 'wearing_necktie': 1})
celeba_young = TensorMap('young', Interpretation.CATEGORICAL, tensor_from_file=celeba_label, channel_map={'no_young': 0, 'young': 1})


def mnist_label_from_hd5(tm: TensorMap, hd5: h5py.File, dependents: Dict = {}) -> np.ndarray:
    one_hot = np.zeros(tm.shape, dtype=np.float32)
    one_hot[int(hd5['mnist_label'][0])] = 1.0
    return one_hot


mnist_label = TensorMap(
    'mnist_label', Interpretation.CATEGORICAL, tensor_from_file=mnist_label_from_hd5,
    channel_map={f'digit_{i}': i for i in range(10)},
)


def mnist_label_as_time_to_event(tm: TensorMap, hd5: h5py.File, dependents: Dict = {}) -> np.ndarray:
    tensor = np.zeros(tm.shape, dtype=np.float32)
    label = float(hd5['mnist_label'][0])
    tensor[0] = 1.0 if np.random.rand() > (label / 10) else 0.0
    tensor[1] = np.random.randint(1, 3650)
    return tensor


mnist_time_to_event = TensorMap(
    'mnist_time_to_event', Interpretation.TIME_TO_EVENT,
    tensor_from_file=mnist_label_as_time_to_event,
)


def mnist_label_as_survival_curve(tm: TensorMap, hd5: h5py.File, dependents: Dict = {}) -> np.ndarray:
    label = float(hd5['mnist_label'][0])
    has_disease = 1.0 if np.random.rand() > (label / 10) else 0.0
    days_follow_up = np.random.randint(1, 3650)

    intervals = int(tm.shape[0] / 2)
    days_per_interval = tm.days_window / intervals
    survival_then_censor = np.zeros(tm.shape, dtype=np.float32)
    for i, day_delta in enumerate(np.arange(0, tm.days_window, days_per_interval)):
        survival_then_censor[i] = float(day_delta < days_follow_up)
        if day_delta <= days_follow_up < day_delta + days_per_interval:
            survival_then_censor[intervals + i] = has_disease
    return survival_then_censor


mnist_survival_curve = TensorMap(
    'mnist_survival_curve', Interpretation.SURVIVAL_CURVE, shape=(50,),
    tensor_from_file=mnist_label_as_survival_curve,
)


def mnist_as_hd5(hd5_folder):
    train, _, _ = load_data('mnist.pkl.gz')
    mnist_images = train[0].reshape((-1, 28, 28, 1))
    if not os.path.exists(hd5_folder):
        os.makedirs(hd5_folder)
    for i, mnist_example in enumerate(mnist_images):
        with h5py.File(os.path.join(hd5_folder, f'{i}.hd5'), 'w') as hd5:
            hd5.create_dataset('mnist_image', data=mnist_example)
            hd5.create_dataset('mnist_label', data=[train[1][i]])
        if (i+1) % 5000 == 0:
            print(f'Wrote {i+1} MNIST images and labels as HD5 files')


def load_data(dataset):
    ''' Loads the dataset
    :param dataset: the path to the dataset (here MNIST)'''
    data_dir, data_file = os.path.split(dataset)
    if data_dir == "" and not os.path.isfile(dataset):
        # Check if dataset is in the data directory.
        new_path = os.path.join("data", dataset)
        if os.path.isfile(new_path) or data_file == 'mnist.pkl.gz':
            dataset = new_path

    if (not os.path.isfile(dataset)) and data_file == 'mnist.pkl.gz':
        from urllib.request import urlretrieve
        origin = ('http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz')
        print('Downloading data from %s' % origin)
        if not os.path.exists(os.path.dirname(dataset)):
            os.makedirs(os.path.dirname(dataset))
        urlretrieve(origin, dataset)

    print('loading data...')
    f = gzip.open(dataset, 'rb')
    if sys.version_info[0] == 3:
        u = pickle._Unpickler(f)
        u.encoding = 'latin1'
        train_set, valid_set, test_set = u.load()
    else:
        train_set, valid_set, test_set = pickle.load(f)
    f.close()

    return train_set, valid_set, test_set
