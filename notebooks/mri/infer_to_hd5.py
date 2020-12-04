import ml4h
import tensorflow_addons as tfa
import tensorflow as tf
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow_addons.optimizers import RectifiedAdam
from tensorflow_addons.layers import InstanceNormalization
from tensorflow.python.data.ops import dataset_ops
from ml4h.tensor_generators import TensorGenerator
import numpy as np
import h5py
import os
import cv2
import sys
import glob
from ml4h.metrics import get_metric_dict
import ml4h.tensormap.ukb.mri
from ml4h.tensormap.general import pad_or_crop_array_to_shape
from ml4h.normalizer import ZeroMeanStd1
import zstandard
import blosc
​
# Test
# Path to a pre-trained file
# model_file = "/tf/models_sax_slices_jamesp_4b_converge_sax_slices_jamesp_4b_converge.h5"
model_file = "/home/pdiachil/sax_slices_jamesp_4b_hyperopted_dropout_pap_dupe.h5"
# Get the config using the TensorMap used to train the model in the first place.
# This allows us to reconstruct models that are not saved using configs.
objects = get_metric_dict([ml4h.tensormap.ukb.mri.cine_segmented_sax_slice_jamesp])
# Silly work-around to reset function pointer to our local instance.
objects['RectifiedAdam'] = RectifiedAdam
# Load the model
model = load_model(model_file, custom_objects=objects)
model.summary()
​
​
import timeit
hd5s = glob.glob('/mnt/disks/annotated-cardiac-tensors-44k/2020-09-21/*.hd5')

start = int(sys.argv[1])
end = int(sys.argv[2])

step = df.shape[0]//10
batch = 5

​
tot = 0
for ppp, hd5 in enumerate(sorted(hd5s)):
    sample_id = hd5.split('/')[-1].replace('.hd5', '')
    if ppp < start:
        continue
    if ppp == end:
        break
    print(f"{sample_id}. Progress: {tot}/{len(hd5s)}")
    try:
        zzz = f'/mnt/disks/annotated-cardiac-tensors-44k/2020-09-21/{sample_id}.hd5'
        x = h5py.File(zzz, 'r')
    except Exception as e:
        print("Failed to open")
        continue
    #
    names = pd.DataFrame({'path': list(x['/ukb_cardiac_mri/'])})
    names = names[names.loc[:,'path'].str.contains('cine_segmented_sax_b')]
    names['slices'] = [int(s.split('_')[-1].replace('b','')) for s in names['path'].values]
    names = names.sort_values(['slices'])
    names = names[~names.path.str.contains('james')]
    # Grab data
    tensor = np.zeros((len(names), 50, 224, 224), dtype=np.float32)
    for f,k in zip(names.path, range(len(names))):
        tensor[k,...] = ZeroMeanStd1().normalize(pad_or_crop_array_to_shape((50, 224, 224), 
            np.moveaxis(x['/ukb_cardiac_mri/'][f]['2/instance_0'][()], 2, 0)))
    #
    #
    start_predict = timeit.default_timer() # Debug timer
    test   = np.zeros((len(names), 50, 224, 224, 17), dtype=np.float32)
    argmax = np.zeros((len(names), 50, 224, 224),     dtype=np.float32)
    #
    #
    for i in range(len(names)):
        tensor_local = np.zeros((50, 224, 224, 4))
        # 
        print(max(i-2, 0), max(i-1, 0), i, min(i+1, len(names)-1))
        if i-2 >= 0:
            tensor_local[:,:,:,0] = tensor[(i-2),...]
        else:
            if -1 >= 0:
                tensor_local[:,:,:,0] = tensor[(i-1),...]
            else:
                tensor_local[:,:,:,0] = tensor[(i),...]
        if i-1 >=0:
            tensor_local[:,:,:,1] = tensor[(i-1),...]
        else:
            tensor_local[:,:,:,1] = tensor[0,...]
        tensor_local[:,:,:,2] = tensor[(i),...]
        if i+1 < len(names):
            # print('i+1 is valid')
            tensor_local[:,:,:,3] = tensor[(i+1),...]
        else:
            tensor_local[:,:,:,3] = tensor[i,...]
        for j in range(50):
            pred = model.predict(tensor_local[j:(j+1), ...])
            # print(pred.shape)
            test[i,j, ...]   = tf.squeeze(pred)
            argmax[i,j, ...] = tf.argmax(pred, axis=-1)
    #
    #
    with h5py.File(f'/home/pdiachil/{str(s)}_inference__argmax.h5', 'w-') as ff:
        # ff.create_dataset("predictions", data=test)
        # ff.create_dataset("data", data=tensor)
        ff.create_dataset("argmax", data=np.void(blosc.compress(tf.argmax(test,axis=-1).numpy().astype(np.uint8).tobytes(), typesize=2, cname='zstd', clevel=9)))
        ff.attrs['shape'] = test.shape
        #
        #
    stop_predict  = timeit.default_timer() # Debug timer
    print('Predict time: ', stop_predict - start_predict) # Debug message
    tot += 1
