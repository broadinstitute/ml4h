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
import ml4h
import tensorflow_addons as tfa
import tensorflow as tf
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow_addons.optimizers import RectifiedAdam
import numpy as np
import h5py
import os
import glob
from ml4h.metrics import get_metric_dict
import ml4h.tensormap.ukb.mri
from ml4h.tensormap.general import pad_or_crop_array_to_shape
from ml4h.normalizer import ZeroMeanStd1
import zstandard
import blosc
import pyarrow
from io import BytesIO
import sys
import timeit

if len(sys.argv) != 3:
    raise Exception('Need to provide 2 parameters: #machines, current machine#')

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

# ---------- From the command line
# number of machines, current machine
files = sorted(glob.glob('/mnt/disks/annotated-cardiac-tensors-44k/2020-09-21/*.hd5'))
print(len(files))
step = len(files)//int(sys.argv[1])
batch = int(sys.argv[2])
if batch != step:
    files = files[(step*(batch)):(step*(batch+1))]
else:
    files = files[(step*(batch)):]
print(len(files))

def prepare_local_tensor(tensor, names):
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
        tensor_local[:,:,:,3] = tensor[(i+1),...]
    else:
        tensor_local[:,:,:,3] = tensor[i,...]
    return tensor_local

# s = ['/mnt/disks/annotated-cardiac-tensors-44k/2020-09-21/1986469.hd5']

tot = 0
for s in files:
    print(f"{s}. Progress: {tot}/{len(files)}")
    try:
        # zzz = f'/mnt/disks/annotated-cardiac-tensors-44k/2020-09-21/{str(s)}.hd5'
        x = h5py.File(s,'r')
    except Exception as e:
        print(f"Failed to open {s}")
        continue
    #
    try:
        names = pd.DataFrame({'path': list(x['/ukb_cardiac_mri/'])})
    except Exception as e:
        print(f'Malformed hdf5 {s}')
        continue
    #
    names = names[names.loc[:,'path'].str.contains('cine_segmented_sax_b')]
    names['slices'] = [int(s.split('_')[-1].replace('b','')) for s in names['path'].values]
    names = names.sort_values(['slices'])
    names = names[~names.path.str.contains('james')]
    #
    if len(names) == 0:
        print(f"Failed to find data for {s}")
        continue
    #
    instances = list(x['/ukb_cardiac_mri/'][names.path.iloc[0]])
    #
    for instance in instances:
        print(f'instance: {instance}')
        names_valid = names[[instance in list(x['/ukb_cardiac_mri/'][p]) for p in names.path]]
        # Grab data
        tensor = np.zeros((len(names_valid), 50, 224, 224), dtype=np.float32)
        for f,k in zip(names_valid.path, range(len(names_valid))):
            tensor[k,...] = ZeroMeanStd1().normalize(pad_or_crop_array_to_shape((50, 224, 224), 
                np.moveaxis(x['/ukb_cardiac_mri/'][f][instance]['instance_0'][()], 2, 0)))
        #
        #
        start_predict = timeit.default_timer() # Debug timer
        test   = np.zeros((len(names_valid), 50, 224, 224, 17), dtype=np.float32)
        # argmax = np.zeros((len(names_valid), 50, 224, 224),     dtype=np.float32)
        #
        #
        for i in range(len(names_valid)):
            tensor_local = prepare_local_tensor(tensor, names_valid)
            for j in range(50):
                pred = model.predict(tensor_local[j:(j+1), ...])
                test[i,j, ...]   = tf.squeeze(pred)
                # argmax[i,j, ...] = tf.argmax(pred, axis=-1)
        #
        #
        filename = os.path.splitext(os.path.split(s)[-1])[0]
        with h5py.File(f'/home/pdiachil/{filename}_inference__argmax.h5', 'a') as ff:
            ff_in = ff.create_group(f'instance_{str(instance)}')
            ff_in.create_dataset("argmax", data=np.void(blosc.compress(tf.argmax(test,axis=-1).numpy().astype(np.uint8).tobytes(), typesize=2, cname='zstd', clevel=9)))
            ff_in.attrs['shape'] = test.shape
            # Hard-core approach to store Parquet as an in-memory view
            buffer = BytesIO()
            names_valid.to_parquet(buffer, engine='pyarrow', compression='zstd')
            ff_in.create_dataset("slices_pq", data=np.void(buffer.getvalue()))
            # Getting data back:
            # pd.read_parquet(BytesIO(buffer), engine='pyarrow')
        #
        #
        stop_predict  = timeit.default_timer() # Debug timer
        print('Predict time: ', stop_predict - start_predict) # Debug message
        tot += 1

