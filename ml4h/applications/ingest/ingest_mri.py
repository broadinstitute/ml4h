import pydicom as dicom
from dicom_parser import Image
from dicom_parser.utils.siemens.csa.header import CsaHeader
import numpy as np
import pandas as pd
import glob
import fastparquet as fp
import blosc
import xxhash
import h5py
import sys


ukbid = 999999999 # In practice we read this from sys.argv
files = glob.glob("bodymri_all_raw_999999999_20201_2_0/*.dcm") # DICOMs for this individual


# The (0029, xxxx) tags are SIEMENS-specific and describe CSA header information.
# Read more here: https://nipy.org/nibabel/dicom/siemens_csa.html
#
# (0029, 1008) [CSA Image Header Type]             OB: 'IMAGE NUM 4 '
# (0029, 1009) [CSA Image Header Version]          OB: '20100114'
# (0029, 1010) [CSA Image Header Info]             OB: Array of 11560 bytes
# (0029, 1018) [CSA Series Header Type]            OB: 'MR'
# (0029, 1019) [CSA Series Header Version]         OB: '20100114'
# (0029, 1020) [CSA Series Header Info]            OB: Array of 80248 bytes

# Extract data for all the DICOMs
dfs = []
for f in files:
    try:
        ds = dicom.read_file(f)
        # image = Image(f) # Use if we are storing the CSA header information
    except Exception as e:
        print(f'Failed to open file {f} with exception: {e}')
        continue
    df = pd.DataFrame(ds.values())
    df[0] = df[0].apply(lambda x: dicom.dataelem.DataElement_from_raw(x) if isinstance(x, dicom.dataelem.RawDataElement) else x)
    # df['tag'] = df[0].apply(lambda x: [hex(x.tag.group), hex(x.tag.elem)]) # If we want to store the tag tuple
    df['name']  = df[0].apply(lambda x: x.name)
    df['value'] = df[0].apply(lambda x: x.value)
    # df = df[['tag','name', 'value']] # If we're storingthe tags
    df = df[['name', 'value']]
    df = df.append({'name': 'pixel_array_shape', 'value': ds.pixel_array.shape}, ignore_index=True)
    # df = df.append({'tag': [9999,9999] ,'name': 'pixel_array_shape', 'value': ds.pixel_array.shape}, ignore_index=True)
    # I don't believe there is anything in the CSA header we would like to have
    # immediately available.
    # csa_header = CsaHeader(image.header.get('CSASeriesHeaderInfo')).parsed
    df2 = df.copy()
    df2 = df2['name'][~df2['name'].str.contains('Private')] # Drop private tags -- mostly SIEMENS information
    df2 = df[df['name'].isin(df2)]
    df2 = df2.set_index(df2.name)
    df2 = df2[['value']]
    df2 = df2.transpose()
    df2['dicom_name'] = f
    df2 = df2.drop(labels='Pixel Data',axis=1)
    df2['pixel_data'] = [ds.pixel_array]
    dfs.append(df2)


sample_manifest = pd.concat(dfs)
# >>> pd.crosstab(sample_manifest['pixel_array_shape'], sample_manifest['Series Number'])
# Series Number      1   2   3   4   5   6   7   8   9   10  11  12  13  14  15  16  17  18  19  20  21  22  23  24
# pixel_array_shape
# (156, 224)          0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0  64  64  64  64
# (162, 224)          0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0  72  72  72  72   0   0   0   0
# (168, 224)         64  64  64  64   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
# (174, 224)          0   0   0   0  44  44  44  44  44  43  44  44  44  44  44  44   0   0   0   0   0   0   0   0
sample_manifest['Series Number']   = sample_manifest['Series Number'].astype(np.int32)
sample_manifest['Instance Number'] = sample_manifest['Instance Number'].astype(np.int32)
sample_manifest = sample_manifest.sort_values(['Series Number', 'Instance Number'])
sample_manifest = sample_manifest.reset_index()


# # Count uniqueness in fields
# for m in sample_manifest.columns[:-3]:
#     counts = len(np.unique(sample_manifest[m].values))
#     if counts > 1:
#         print(m,counts)

pixel_data = sample_manifest['pixel_data']
pixel_data_shape = sample_manifest['pixel_array_shape']
sample_manifest = sample_manifest.drop(['pixel_data','pixel_array_shape'],axis=1)

# Recode edge cases that are stored as lists
sample_manifest = sample_manifest.drop(['Image Type'],axis=1)
sample_manifest["Patient's Name"] = sample_manifest["Patient's Name"].astype(str)
sample_manifest['Sequence Variant'] = [','.join(list(s)) for s in sample_manifest['Sequence Variant'].values]
patient_position = \
pd.DataFrame([np.array(list(s),dtype=np.float32) for s in sample_manifest['Image Position (Patient)'].values],
    columns=['image_position_x','image_position_y','image_position_z'])
sample_manifest = sample_manifest.drop(['Image Position (Patient)'], axis=1)

patient_orientation = \
pd.DataFrame([np.array(list(s),dtype=np.float32) for s in sample_manifest['Image Orientation (Patient)'].values],
    columns=['image_orientation_row_x','image_orientation_row_y','image_orientation_row_z',
    'image_orientation_col_x','image_orientation_col_y','image_orientation_col_z'])
sample_manifest = sample_manifest.drop(['Image Orientation (Patient)'], axis=1)

pixel_spacing = \
pd.DataFrame([np.array(list(s),dtype=np.float32) for s in sample_manifest['Pixel Spacing'].values],
    columns=['row_pixel_spacing_mm','col_pixel_spacing_mm'])
sample_manifest = sample_manifest.drop(['Pixel Spacing'], axis=1)

acquisition_matrix = \
pd.DataFrame([np.array(list(s),dtype=np.float32) for s in sample_manifest['Acquisition Matrix'].values],
    columns=['acquisition_matrix_freq_rows','acquisition_matrix_freq_cols',
             'acquisition_matrix_phase_rows','acquisition_matrix_phase_cols'])
sample_manifest = sample_manifest.drop(['Acquisition Matrix'], axis=1)

# Add UKBID as a categorical to the manifest
sample_manifest['ukbid'] = ukbid
sample_manifest['ukbid'] = pd.Categorical(sample_manifest['ukbid'])

# Concat new values
sample_manifest = pd.concat([sample_manifest,patient_position,patient_orientation,pixel_spacing,acquisition_matrix,],axis=1)

# Recast object data to proper primitive types
sample_manifest['Acquisition Date'] = pd.to_datetime(sample_manifest['Acquisition Date'])
sample_manifest['Acquisition Number'] = sample_manifest['Acquisition Number'].astype(np.int32)
sample_manifest['Bits Allocated'] = sample_manifest['Bits Allocated'].astype(np.int8)
sample_manifest['Bits Stored'] = sample_manifest['Bits Stored'].astype(np.int8)
sample_manifest['Columns'] = sample_manifest['Columns'].astype(np.int16)
sample_manifest['Content Date'] = pd.to_datetime(sample_manifest['Content Date'])
sample_manifest['Device Serial Number'] = sample_manifest['Device Serial Number'].astype(np.int32)
sample_manifest['Echo Number(s)'] = sample_manifest['Echo Number(s)'].astype(np.int16)
sample_manifest['Echo Time'] = sample_manifest['Echo Time'].astype(np.float32)
sample_manifest['Echo Train Length'] = sample_manifest['Echo Train Length'].astype(np.int8)
sample_manifest['Flip Angle'] = sample_manifest['Flip Angle'].astype(np.float32)
sample_manifest['High Bit'] = sample_manifest['High Bit'].astype(np.int16)
sample_manifest['Imaging Frequency'] = sample_manifest['Imaging Frequency'].astype(np.float32)
sample_manifest['Instance Creation Date'] = pd.to_datetime(sample_manifest['Instance Creation Date'])
sample_manifest['Instance Number'] = sample_manifest['Instance Number'].astype(np.int8)
sample_manifest['Largest Image Pixel Value'] = sample_manifest['Largest Image Pixel Value'].astype(np.int16)
sample_manifest['Magnetic Field Strength'] = sample_manifest['Magnetic Field Strength'].astype(np.float32)
sample_manifest['Number of Averages'] = sample_manifest['Number of Averages'].astype(np.float32)
sample_manifest['Number of Phase Encoding Steps'] = sample_manifest['Number of Phase Encoding Steps'].astype(np.int16)
sample_manifest["Patient's Birth Date"] = pd.to_datetime(sample_manifest["Patient's Birth Date"])
sample_manifest["Patient's Size"] = sample_manifest["Patient's Size"].astype(np.float32)
sample_manifest["Patient's Weight"] = sample_manifest["Patient's Weight"].astype(np.float32)
sample_manifest['Percent Phase Field of View'] = sample_manifest['Percent Phase Field of View'].astype(np.float32)
sample_manifest['Percent Sampling'] = sample_manifest['Percent Sampling'].astype(np.float32)
sample_manifest['Performed Procedure Step Start Date'] = pd.to_datetime(sample_manifest['Performed Procedure Step Start Date'])
sample_manifest['Pixel Bandwidth'] = sample_manifest['Pixel Bandwidth'].astype(np.float32)
sample_manifest['Pixel Representation'] = sample_manifest['Pixel Representation'].astype(np.float32)
sample_manifest['Repetition Time'] = sample_manifest['Repetition Time'].astype(np.float32)
sample_manifest['Rows'] = sample_manifest['Rows'].astype(np.int16)
sample_manifest['SAR'] = sample_manifest['SAR'].astype(np.float32)
sample_manifest['Samples per Pixel'] = sample_manifest['Samples per Pixel'].astype(np.float32)
sample_manifest['Series Date'] = pd.to_datetime(sample_manifest['Series Date'])
sample_manifest['Series Number'] = sample_manifest['Series Number'].astype(np.int16)
sample_manifest['Slice Location'] = sample_manifest['Slice Location'].astype(np.float32)
sample_manifest['Slice Thickness'] = sample_manifest['Slice Thickness'].astype(np.float32)
sample_manifest['Smallest Image Pixel Value'] = sample_manifest['Smallest Image Pixel Value'].astype(np.int16)
sample_manifest['Study Date'] = pd.to_datetime(sample_manifest['Study Date'])
sample_manifest['Study ID'] = sample_manifest['Study ID'].astype(np.int32)
sample_manifest['Slice Thickness'] = sample_manifest['Slice Thickness'].astype(np.float32)
sample_manifest['Window Width'] = sample_manifest['Window Width'].astype(np.float32)
sample_manifest['dB/dt'] = sample_manifest['dB/dt'].astype(np.float32)

# Reformat column names to be all lower case and replace spaces with underscores
sample_manifest.columns = [s.lower().replace(' ','_') for s in sample_manifest.columns.values]


# Store meta data
fp.write('body_mri_manifest',
    data=sample_manifest,
    row_group_offsets=10000,
    compression="zstd",
    file_scheme='hive',
    partition_on=['ukbid'],
    append=True)


# Open HDF5 for storing the tensors
try:
    f = h5py.File(f"{ukbid}.h5","w-")
except Exception as e:
    raise Exception(e)


# Generate stacks of 2D images into 3D tensors
for s in np.unique(sample_manifest['Series Number']):
    t = np.stack(pixel_data[sample_manifest.loc[sample_manifest['Series Number']==s].index],axis=2)
    hash_uncompressed = xxhash.xxh128_digest(t)
    hash_compressed = xxhash.xxh128_digest(blosc.compress(t, typesize=2, cname='zstd', clevel=9))
    compress = blosc.compress(t.tobytes(), typesize=2, cname='zstd', clevel=9)
    decompressed = np.frombuffer(blosc.decompress(compress),dtype=np.uint16).reshape(t.shape)
    assert(xxhash.xxh128_digest(decompressed) == hash_uncompressed)
    dset = f.create_dataset(f"/series/{s}", data=np.void(compress))
    # Store meta data
    dset.attrs['shape'] = t.shape
    dset.attrs['hash_compressed'] = np.void(hash_compressed)
    dset.attrs['hash_uncompressed'] = np.void(hash_uncompressed)
    
# Done