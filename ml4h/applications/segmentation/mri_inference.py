import ml4h
import tensorflow_addons as tfa
import tensorflow as tf
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow_addons.optimizers import RectifiedAdam
# from ml4h.metrics import get_metric_dict
# import ml4h.tensormap.ukb.mri
# from ml4h.tensormap.general import pad_or_crop_array_to_shape
# from ml4h.normalizer import ZeroMeanStd1
from tensorflow.python.data.ops import dataset_ops
from ml4h.tensor_generators import TensorGenerator
import numpy as np
import h5py
import os
import cv2
# import glob # Use glob if a manifest is not available
# import fastparquet as fp
import sys
import glob
# debug
from ml4h.metrics import get_metric_dict
import ml4h.tensormap.ukb.mri


# temp utils
class DataGenerator(tf.keras.utils.Sequence):
    """Workaround for using infinite ML4H generators with Keras/Tensorflow 
    generators. The ML4H TensorGenerator is very brittle: use at your own
    risk!
    Args:
        files: Files operated on by TensorGenerator.
        tensor_generator: ML4h TensorGenerator instance.
    """
    def __init__(self, files, tensor_generator: TensorGenerator):
        self.files = files
        self.generator = tensor_generator
        self.offset = 0
        self.limit = len(files)
    #
    def __len__(self):
        return self.limit
    #
    def __getitem__(self, index):
        self.offset += 1
        x, y, _, b = next(self.generator)
        return x, y, [None]
    #
    def on_epoch_end(self):
        self.offset = 0


def predict_segmentation(model: tf.keras.models.Model,
        generator):
    """Make inference using a generator by iteratively retrieving data. This
    approach is only efficient in a batch-wise setting. All preprocessing must
    be performed inside the generator.

    Args:
        model (tf.keras.models.Model): Input Tensorflow model
        generator: Generator to draw data from.
    """
    if isinstance(generator, dataset_ops.DatasetV2):
        pass
    elif isinstance(generator, TensorGenerator):
        files = np.array([p.paths for p in generator.path_iters]).flatten()
        print(files)
        generator = DataGenerator(files, generator)
    else: # yolo
        pass

    # Make inference
    x, y = next(generator)
    start_predict = timeit.default_timer() # Debug timer
    predictions   = model.predict(x) # Predict all (d, 224, 224) tensors at once
    stop_predict  = timeit.default_timer() # Debug timer
    # Ev


# Test
# Path to a pre-trained file
model_file = "/tf/models_sax_slices_jamesp_4b_converge_sax_slices_jamesp_4b_converge.h5"
# Get the config using the TensorMap used to train the model in the first place.
# This allows us to reconstruct models that are not saved using configs.
objects = get_metric_dict([ml4h.tensormap.ukb.mri.cine_segmented_sax_slice_jamesp])
# Silly work-around to reset function pointer to our local instance.
objects['RectifiedAdam'] = RectifiedAdam
# Load the model
model = load_model(model_file, custom_objects=objects)
model.summary()


files = glob.glob('/mnt/disks/annotated-cardiac-tensors-44k/2020-09-21/*.hd5')[0:10]


generate_test = TensorGenerator(
    1, 
    [ml4h.tensormap.ukb.mri.sax_slices_jamesp_4b], 
    [ml4h.tensormap.ukb.mri.cine_segmented_sax_slice_jamesp], 
    paths=files,
    num_workers=1,
    cache_size=0, 
    keep_paths=True, 
    mixup=0,
)




# Command line helper for divide-and-conquer approach to inferring
# millions of segmentations. See grouping information below
grouping = 0
if len(sys.argv) == 1:
    print('must provide argument')
    exit(1)
else:
    grouping = int(sys.argv[1]) # No validty checks are made


# Manifest file. This tells us where our files are located how they map back to the DICOMs
manifest = fp.ParquetFile('path_to_manifest.pq')
manifest = manifest.to_pandas()
# Reference manifest content:
#
# >>> manifest.iloc[0]
# sample_id                                                        <HIDDEN>
# field_id                                                            20209
# instance                                                                2
# index                                                                   0
# zip_file                                                 <HIDDEN>_2_0.zip
# dicom_file                                                       <HIDDEN>
# series                                              CINE_segmented_SAX_b3
# date                                                           2014-07-25
# instance_number                                                        42
# overlay_text                                                    NoOverlay
# overlay_fraction                                                        0
# overlay_rows                                                            0
# overlay_cols                                                            0
# rows                                                                  208
# cols                                                                  210
# image_x                                                            114.71
# image_y                                                            -14.38
# image_z                                                            271.48
# px_height_mm                                                          1.9
# px_width_mm                                                           1.9
# slice_thickness_mm                                                      8
# series_number                                                          12
# acquisition_number                                                      1
# device_serial_number                                                41754
# station_name                                                     AWP41754
# software_versions                                            syngo MR D13
# echo_time                                                             1.1
# nominal_interval                                                     1042
# slice_location                                                   -14.1349
# trigger_time                                                       854.44


# Path to a pre-trained model file
model_file = "path_to_model.h5"
# These next two lines assumes the model was generated using ML4H logic.
# 
# Get the config using the TensorMap used to train the model in the first place.
# This allows us to reconstruct models that are not saved using configs.
# In this example use-case we built the model using the `cine_segmented_sax_slice_jamesp`
# tensormap.
objects = get_metric_dict([ml4h.tensormap.ukb.mri.cine_segmented_sax_slice_jamesp])
# Silly work-around to reset function pointer to our local address.
# This shoud be considered a bug.
objects['RectifiedAdam'] = RectifiedAdam

# Load the model
m = load_model(model_file, custom_objects=objects)
m.summary() # Optional verbosity
# Example output
# ... truncated ...
# __________________________________________________________________________________________________
# output_cine_segmented_sax_slice (None, 224, 224, 12) 1164        concatenate_33[0][0]
# ==================================================================================================
# Total params: 5,404,204
# Trainable params: 5,404,204
# Non-trainable params: 0
# __________________________________________________________________________________________________

# Hackery to split the workload into 20 even bins
files = pd.read_csv('path_to_groupings.txt')
files = files[files.group==grouping] # Select the bucket of interest
# Example output where the first 5 HDF5s are assigned to group index 0
# 
# $ head path_to_groupings.txt
# /mnt/disks/annotated-cardiac-tensors-44k/2020-09-21/<HIDDEN>.hd5,0
# /mnt/disks/annotated-cardiac-tensors-44k/2020-09-21/<HIDDEN>.hd5,0
# /mnt/disks/annotated-cardiac-tensors-44k/2020-09-21/<HIDDEN>.hd5,0
# /mnt/disks/annotated-cardiac-tensors-44k/2020-09-21/<HIDDEN>.hd5,0
# /mnt/disks/annotated-cardiac-tensors-44k/2020-09-21/<HIDDEN>.hd5,0
# /mnt/disks/annotated-cardiac-tensors-44k/2020-09-21/<HIDDEN>.hd5,0

# Prepares names
ids = [int(os.path.split(f)[-1].split('.')[0]) for f in files.file.values]
manifest = manifest[manifest.sample_id.isin(ids)]
manifest['series_value'] = [int(m[20:]) for m in manifest.series.values]

# Prepare the output path for images
import timeit
from datetime import datetime
output_path = os.path.join('output_path', datetime.now().strftime('%Y-%m-%d'))
if not os.path.exists(output_path):
    os.makedirs(output_path)


current_file_count = 0 # Debugging counter

for target_file in files.file.values: # Iterate over our files
    print(f"Current file: {current_file_count}/{len(files)} --- {target_file}")
    # Parse the UKB identifier and make an output path per subject
    ukb_id = os.path.split(target_file)[-1].split('.')[0]
    local_output_path = os.path.join(output_path, ukb_id)
    if not os.path.exists(local_output_path):
        os.makedirs(local_output_path)
    
    # Open HDF5 file handle in read-only mode
    f = h5py.File(target_file,'r')
    
    try:
        # Load the data frame of interest and string parse the b index
        cardiac_keys   = pd.DataFrame(f['/ukb_cardiac_mri/'].keys())
        cardiac_slices = cardiac_keys[cardiac_keys.iloc[:,0].str.contains('cine_segmented_sax_b')]
        cardiac_slices_order = [int(c.split('_')[-1][1:1000]) for c in cardiac_slices[0].values]
        cardiac_slices = cardiac_slices.iloc[np.argsort(cardiac_slices_order)]
    except Exception as e:
        print(f"Failed to run {target_file}")
        continue # Do not fail
    
    #
    start = timeit.default_timer() # Debug timer
    
    # make_assertions = True # Assert copies
    # DataGenerator
    for instance in cardiac_slices.iloc[:,0]:
        instances = list(f['/ukb_cardiac_mri/'][instance]) # List the instances availabe
        for i in instances: # Iterate over instances
            start_collect = timeit.default_timer() # Debug timer
            try:
                # Load the data frame of interest
                dat = f['/ukb_cardiac_mri/'][instance][i]['instance_0'][()]
            except Exception as e:
                print(f'Failed to load instance: {instance}. Specfically, /ukb_cardiac_mri/{instance}/2/instance_0')
                continue # Do not fail
            
            # Aggregate data and their corresponding DICOM names
            dat_all = []
            dicom_names = []
            inner_manifest = manifest[(manifest['sample_id']==int(ukb_id))&(manifest['instance']==int(i))&(manifest['series_value']==int(instance[20:]))]
            # Loop over the last channel in our tensor. This corresponds to the time component
            for d in range(dat.shape[-1]):
                try:
                    # Store the DICOM name
                    dicom_names.append(str(inner_manifest.loc[manifest['instance_number']==int(d+1)].dicom_file.values[0]))
                except Exception as e:
                    print(f'cant find dicom name for: {ukb_id}, {i}, {d}, {instance[20:]}')
                    continue
                # Reshape tensor-slice at index d to (224, 224, 1) and store in tensor
                tensor = pad_or_crop_array_to_shape((224, 224, 1), dat[..., d])
                # Normalize as in the original model
                tensor = ZeroMeanStd1().normalize(tensor)
                # Store tensor
                dat_all.append(tensor)
            # End data preparation loop

            stop_collect = timeit.default_timer() # Debug timer
            print('Tensorize time: ', stop_collect - start_collect) # Debug message
            
            # Make predictions
            start_predict = timeit.default_timer() # Debug timer
            predictions = m.predict(np.array(dat_all)) # Predict all (224, 224, d) tensors at once
            stop_predict = timeit.default_timer() # Debug timer
            print('Predict time: ', stop_predict - start_predict) # Debug message
            
            # Aux PNG
            # Convert predictions into valid image segmentations
            for p,n in zip(predictions,dicom_names):
                # Argmax over channels: this is equivalent of selecting the one-hot-encoded channel
                # with the largest probability
                maxpred = np.argmax(p,axis=2)
                # Copy predictions to all 3 channels
                expanded = np.repeat(maxpred[:, :, np.newaxis], 3, axis=2)

                # # Make assertions that all channels are identical
                # if make_assertions:
                #     assert(np.all(expanded[:,:,0]==expanded[:,:,1]) == True)
                #     assert(np.all(expanded[:,:,0]==expanded[:,:,2]) == True)
                #     assert(np.all(expanded[:,:,1]==expanded[:,:,2]) == True)
                
                filename = n + ".png.mask.png"
                try:
                    # Write file to disk
                    capture = cv2.imwrite(os.path.join(local_output_path, filename), expanded)
                except Exception as e:
                    print(f"Failed to save image: {os.path.join(local_output_path, filename)}")
                    raise Exception(e) # Do fail: irrecoverable error
            # End predictions loop
        stop = timeit.default_timer() # Debug timer
        print('Time: ', stop - start) # Debug message
    current_file_count += 1 # Debug counter
