import pydicom as dicom
# Uncomment if you want to store the SIEMENS CSA header information
# from dicom_parser import Image
# from dicom_parser.utils.siemens.csa.header import CsaHeader
import numpy as np
import pandas as pd
import glob
from typing import List, Tuple, Optional
import time
import fastparquet as fp
import blosc
import xxhash # Checksums
import h5py
import os
import logging
import zipfile # Read Zip archives
import fnmatch # Simple pattern matching
import zstandard # Silently required by Parquet and Blosc
from io import BytesIO # Read DICOMs from byte streams

# multiprocessing
from multiprocessing import Pool, cpu_count
from functools import partial


SERIES_TO_SAVE = [5, 6, 7]


def _sample_id_from_path(path: str) -> int:
    """assumes naming format is /path/to/file/{sample_id}_other_stuff"""
    return int(os.path.basename(path).split('_')[0])


def _process_file(path: str, destination: str) -> Tuple[str, Optional[str]]:
    sample_id = _sample_id_from_path(path)
    try:
        ingest_mri_dicoms_zipped(sample_id, path, destination)
        return path, None
    except Exception as e:
        return path, str(e)



def multiprocess_ingest(
    files: List[str],
    destination: str,
):
    print(f'Beginning ingestion of {len(files)} MRIs.')
    start = time.time()
    process_file = partial(_process_file, destination=destination)
    errors = {}
    with Pool(cpu_count()) as pool:
        for i, (path, error) in enumerate(pool.imap_unordered(process_file, files)):
            print(f'{i / len(files):.2%}', end='\r')
            if error is not None:
                errors[path] = error
    print()
    delta = time.time() - start
    print(f'Ingestion took {delta:.1f} seconds at {delta / len(files):.1f} s/file')
    return errors


def ingest_mri_dicoms_zipped(
    sample_id: str,
    file: str,
    destination: str,
    output_name: str = None,
    in_memory: bool = True,
    save_dicoms: bool = False,
):
    """Ingest UK Biobank MRI DICOMs from the provided Zip archives comprising of images
    for each individual.

    Args:
        sample_id (str): Sample identifier: this is stored together with the meta data
        file (str): Input path to target Zip archive
        destination (str): Output path
        output_name (str, optional): Output name prefix for HDF5 and Parquet files. Defaults to None.
        in_memory (bool, optional): Unzip and process data entirely in memory resulting in considerably faster processing speeds and less disk usage. Defaults to True.
        save_dicoms (bool, optional): Keep DICOM files in the destination directory after ingestion is complete. Only applicable when `in_memory` is `False`. Defaults to False.

    Example usage:

    >>> base_path = 'bodymri_all_raw_1111111111_20201_2_0'
    >>> ingest_mri_dicoms_zipped(
        sample_id=1111111111,
        file=os.path.join(base_path,'bodymri_all_raw_1111111111_20201_2_0.zip'),
        destination=base_path,
        in_memory=True,
        save_dicoms=False,
        output_name = 'bodymri_1111111111'
    ) # Two output files `bodymri_1111111111.h5` and `bodymri_1111111111.pq`

    This function is embarassingly parallel and as such can be
    wrapper in a function and passed to multiprocessing frameworks such as `multiprocessing`.

    """
    # Make sure the destination folder exists: otherwise create it.
    if not os.path.exists(destination):
        os.makedirs(destination, exist_ok=True)

    zfile = zipfile.ZipFile(file, 'r')

    # Loop over names in the archive
    dicom_files = []
    for name in zfile.namelist():
        # Only open DICOM files in the zip archive
        if fnmatch.fnmatch(name, '*.dcm'):
            dicom_files.append(name)

    if in_memory:
        data = {name: zfile.read(name) for name in dicom_files}
        ingest_mri_dicoms(sample_id, data_dictionary=data, output_name=output_name, destination=destination)
    else:
        zfile.extractall(destination)

        dicoms = glob.glob(os.path.join(destination,"*.dcm"))
        ingest_mri_dicoms(sample_id, files=dicoms, output_name=output_name, destination=destination)

        if save_dicoms == False:
            for file in zfile.namelist():
                print(f"deleting {os.path.join(destination,file)}")
                os.remove(os.path.join(destination,file))


def _ingest_mri_dicoms_wrapper(file, partial=None):
    if partial is None:
        try:
            ds = dicom.read_file(file) # Read the DICOM
            # image = Image(f) # Use if we are storing the CSA header information
        except Exception as e:
            print(f'Failed {file} with exception: {e}')
            return False, None
    else:
        try:
            ds = dicom.dcmread(BytesIO(partial))
        except Exception as e:
            print(f'Failed reading from blob with exception: {e}')
            return False, None

    df = pd.DataFrame(ds.values())
    # Extract out pydicom elements
    df[0] = df[0].apply(lambda x: dicom.dataelem.DataElement_from_raw(x) if isinstance(x, dicom.dataelem.RawDataElement) else x)
    # If we're storing the associated DICOM tags
    # df['tag'] = df[0].apply(lambda x: [hex(x.tag.group), hex(x.tag.elem)])
    df['name']  = df[0].apply(lambda x: x.name)
    df['value'] = df[0].apply(lambda x: x.value)
    # df = df[['tag','name', 'value']]
    df = df[['name', 'value']]
    df = df.append({'name': 'pixel_array_shape', 'value': ds.pixel_array.shape}, ignore_index=True)
    # I don't believe there is anything in the CSA header we would like to have
    # immediately available.
    # csa_header = CsaHeader(image.header.get('CSASeriesHeaderInfo')).parsed
    df2 = df.copy()
    df2 = df2['name'][~df2['name'].str.contains('Private')] # Drop private tags -- mostly SIEMENS information
    df2 = df[df['name'].isin(df2)]
    df2 = df2.set_index(df2.name)
    df2 = df2[['value']]
    df2 = df2.transpose() # Transpose to row-centric orientation
    df2['dicom_name'] = file # Store the DICOM name including file extension
    df2 = df2.drop(labels='Pixel Data',axis=1) # Drop pixel data - handled separately
    df2['pixel_data'] = [ds.pixel_array] # Store pixel array data
    return True, df2


def ingest_mri_dicoms(
    sample_id: str,
    files: str = None,
    data_dictionary: dict = None,
    output_name: str = None,
    destination: str = None,
):
    file_extension = '.h5'
    if output_name is None:
        output_name = str(sample_id)
    else:
        fn, fe = os.path.splitext(output_name)
        if len(fe) != 0:
            if fe.lower() not in ['.hd5', '.h5', '.hdf5']:
                output_name = fn
                file_extension = '.h5'
            else:
                output_name = fn
                file_extension = fe

    if destination is None:
        destination = ''

    # Example additional meta information from the UK Biobank MRI DICOMs:
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
    if files is not None:
        for f in files: # Iterate over files
            # print(f'Parsing: {f}')
            status, df = _ingest_mri_dicoms_wrapper(f)
            if status == True:
                dfs.append(df)
    elif data_dictionary is not None:
        for k,v in data_dictionary.items():
            status, df = _ingest_mri_dicoms_wrapper(k, partial=v)
            if status == True:
                dfs.append(df)


    # Concatenate all row-centric meta data together into a Pandas DataFrame.
    sample_manifest = pd.concat(dfs)
    # Example series-pixel shape relationship for the UK Biobank whole-body MRI DICOMs:
    #
    # >>> pd.crosstab(sample_manifest['pixel_array_shape'], sample_manifest['Series Number'])
    # Series Number      1   2   3   4   5   6   7   8   9   10  11  12  13  14  15  16  17  18  19  20  21  22  23  24
    # pixel_array_shape
    # (156, 224)          0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0  64  64  64  64
    # (162, 224)          0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0  72  72  72  72   0   0   0   0
    # (168, 224)         64  64  64  64   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0   0
    # (174, 224)          0   0   0   0  44  44  44  44  44  43  44  44  44  44  44  44   0   0   0   0   0   0   0   0

    # Cast series and instance number as integers and then sort on those keys --- first by
    # series number and then by instance number.
    sample_manifest['Series Number']   = sample_manifest['Series Number'].astype(np.int32, errors='ignore')
    sample_manifest['Instance Number'] = sample_manifest['Instance Number'].astype(np.int32, errors='ignore')
    sample_manifest = sample_manifest.sort_values(['Series Number', 'Instance Number'])
    sample_manifest = sample_manifest.reset_index()

    # Store pixel data separately and then drop from data frame
    pixel_data = sample_manifest['pixel_data']
    # pixel_data_shape = sample_manifest['pixel_array_shape']
    sample_manifest = sample_manifest.drop(['pixel_data','pixel_array_shape'],axis=1)

    # Recode edge cases
    sample_manifest = sample_manifest.drop(['Image Type'],axis=1)
    sample_manifest['Sequence Variant'] = [','.join(list(s)) for s in sample_manifest['Sequence Variant'].values]
    patient_position = \
    pd.DataFrame(
        [np.array(list(s),dtype=np.float32) for s in sample_manifest['Image Position (Patient)'].values],
        columns=['image_position_x','image_position_y','image_position_z'],
    )
    sample_manifest = sample_manifest.drop(['Image Position (Patient)'], axis=1)

    patient_orientation = \
    pd.DataFrame(
        [np.array(list(s),dtype=np.float32) for s in sample_manifest['Image Orientation (Patient)'].values],
        columns=[
            'image_orientation_row_x','image_orientation_row_y','image_orientation_row_z',
            'image_orientation_col_x','image_orientation_col_y','image_orientation_col_z',
        ],
    )
    sample_manifest = sample_manifest.drop(['Image Orientation (Patient)'], axis=1)

    pixel_spacing = \
    pd.DataFrame(
        [np.array(list(s),dtype=np.float32) for s in sample_manifest['Pixel Spacing'].values],
        columns=['row_pixel_spacing_mm','col_pixel_spacing_mm'],
    )
    sample_manifest = sample_manifest.drop(['Pixel Spacing'], axis=1)

    acquisition_matrix = \
    pd.DataFrame(
        [np.array(list(s),dtype=np.float32) for s in sample_manifest['Acquisition Matrix'].values],
        columns=[
            'acquisition_matrix_freq_rows','acquisition_matrix_freq_cols',
            'acquisition_matrix_phase_rows','acquisition_matrix_phase_cols',
        ],
    )
    sample_manifest = sample_manifest.drop(['Acquisition Matrix'], axis=1)

    # Add UKBID as a categorical to the manifest
    sample_manifest['ukbid'] = sample_id
    sample_manifest['ukbid'] = pd.Categorical(sample_manifest['ukbid'])

    # Concat new values
    sample_manifest = pd.concat([sample_manifest,patient_position,patient_orientation,pixel_spacing,acquisition_matrix],axis=1)

    # Recast --- this is *unsafe* in the general case. There are no guarantees that these fields
    # always exist.
    sample_manifest['Acquisition Date'] = pd.to_datetime(sample_manifest['Acquisition Date'])
    sample_manifest['Acquisition Number'] = sample_manifest['Acquisition Number'].astype(np.int32, errors='ignore')
    sample_manifest['Bits Allocated'] = sample_manifest['Bits Allocated'].astype(np.int8, errors='ignore')
    sample_manifest['Bits Stored'] = sample_manifest['Bits Stored'].astype(np.int8, errors='ignore')
    sample_manifest['Columns'] = sample_manifest['Columns'].astype(np.int16, errors='ignore')
    sample_manifest['Content Date'] = pd.to_datetime(sample_manifest['Content Date'])
    sample_manifest['Device Serial Number'] = sample_manifest['Device Serial Number'].astype(np.int32, errors='ignore')
    sample_manifest['Echo Number(s)'] = sample_manifest['Echo Number(s)'].astype(np.int16, errors='ignore')
    sample_manifest['Echo Time'] = sample_manifest['Echo Time'].astype(np.float32, errors='ignore')
    sample_manifest['Echo Train Length'] = sample_manifest['Echo Train Length'].astype(np.int8, errors='ignore')
    sample_manifest['Flip Angle'] = sample_manifest['Flip Angle'].astype(np.float32, errors='ignore')
    sample_manifest['High Bit'] = sample_manifest['High Bit'].astype(np.int16, errors='ignore')
    sample_manifest['Imaging Frequency'] = sample_manifest['Imaging Frequency'].astype(np.float32, errors='ignore')
    sample_manifest['Instance Creation Date'] = pd.to_datetime(sample_manifest['Instance Creation Date'])
    sample_manifest['Instance Number'] = sample_manifest['Instance Number'].astype(np.int8, errors='ignore')
    sample_manifest['Largest Image Pixel Value'] = sample_manifest['Largest Image Pixel Value'].astype(np.int16, errors='ignore')
    sample_manifest['Magnetic Field Strength'] = sample_manifest['Magnetic Field Strength'].astype(np.float32, errors='ignore')
    sample_manifest['Number of Averages'] = sample_manifest['Number of Averages'].astype(np.float32, errors='ignore')
    sample_manifest['Number of Phase Encoding Steps'] = sample_manifest['Number of Phase Encoding Steps'].astype(np.int16, errors='ignore')
    sample_manifest["Patient's Birth Date"] = pd.to_datetime(sample_manifest["Patient's Birth Date"])
    sample_manifest["Patient's Size"] = sample_manifest["Patient's Size"].astype(np.float32, errors='ignore')
    sample_manifest["Patient's Weight"] = sample_manifest["Patient's Weight"].astype(np.float32, errors='ignore')
    sample_manifest['Percent Phase Field of View'] = sample_manifest['Percent Phase Field of View'].astype(np.float32, errors='ignore')
    sample_manifest['Percent Sampling'] = sample_manifest['Percent Sampling'].astype(np.float32, errors='ignore')
    sample_manifest['Performed Procedure Step Start Date'] = pd.to_datetime(sample_manifest['Performed Procedure Step Start Date'])
    sample_manifest['Pixel Bandwidth'] = sample_manifest['Pixel Bandwidth'].astype(np.float32, errors='ignore')
    sample_manifest['Pixel Representation'] = sample_manifest['Pixel Representation'].astype(np.float32, errors='ignore')
    sample_manifest['Repetition Time'] = sample_manifest['Repetition Time'].astype(np.float32, errors='ignore')
    sample_manifest['Rows'] = sample_manifest['Rows'].astype(np.int16, errors='ignore')
    sample_manifest['SAR'] = sample_manifest['SAR'].astype(np.float32, errors='ignore')
    sample_manifest['Samples per Pixel'] = sample_manifest['Samples per Pixel'].astype(np.float32, errors='ignore')
    sample_manifest['Series Date'] = pd.to_datetime(sample_manifest['Series Date'])
    sample_manifest['Series Number'] = sample_manifest['Series Number'].astype(np.int16, errors='ignore')
    sample_manifest['Slice Location'] = sample_manifest['Slice Location'].astype(np.float32, errors='ignore')
    sample_manifest['Slice Thickness'] = sample_manifest['Slice Thickness'].astype(np.float32, errors='ignore')
    sample_manifest['Smallest Image Pixel Value'] = sample_manifest['Smallest Image Pixel Value'].astype(np.int16, errors='ignore')
    sample_manifest['Study Date'] = pd.to_datetime(sample_manifest['Study Date'])
    sample_manifest['Study ID'] = sample_manifest['Study ID'].astype(np.int32, errors='ignore')
    sample_manifest['Slice Thickness'] = sample_manifest['Slice Thickness'].astype(np.float32, errors='ignore')
    sample_manifest['Window Width'] = sample_manifest['Window Width'].astype(np.float32, errors='ignore')
    sample_manifest['dB/dt'] = sample_manifest['dB/dt'].astype(np.float32, errors='ignore')

    # Reformat column names to be all lower case and replace spaces with underscores
    sample_manifest.columns = [s.lower().replace(' ','_').replace("'", "") for s in sample_manifest.columns.values]

    # Store meta data
    object_columns = sample_manifest.select_dtypes('object')  # all columns where we haven't handled the datatype must be converted to strings
    for col in object_columns:
        sample_manifest[col] = sample_manifest[col].astype('str')
    sample_manifest.to_parquet(os.path.join(destination, f"{output_name}.pq"), compression='zstd')

    # Open HDF5 for storing the tensors
    series = set(SERIES_TO_SAVE).intersection(sample_manifest['series_number'])
    if not series:
        raise ValueError('No series to save.')
    with h5py.File(os.path.join(destination,f"{output_name + file_extension}"),"w-") as f:
        # Generate stacks of 2D images into 3D tensors
        for s in series:
            t = np.stack(pixel_data[sample_manifest.loc[sample_manifest['series_number']==s].index],axis=2)
            compressed_data   = blosc.compress(t.tobytes(), typesize=2, cname='zstd', clevel=9)
            hash_uncompressed = xxhash.xxh128_digest(t)
            hash_compressed   = xxhash.xxh128_digest(compressed_data)
            decompressed = np.frombuffer(blosc.decompress(compressed_data),dtype=np.uint16).reshape(t.shape)
            assert(xxhash.xxh128_digest(decompressed) == hash_uncompressed)
            dset = f.create_dataset(f"/series/{s}", data=np.void(compressed_data))
            # Store meta data:
            # 1) Shape of the original tensor
            # 2) Hash of the compressed data
            # 3) Hash of the uncompressed data
            dset.attrs['shape'] = t.shape
            dset.attrs['hash_compressed']   = np.void(hash_compressed)
            dset.attrs['hash_uncompressed'] = np.void(hash_uncompressed)


def read_compressed(data_set: h5py.Dataset):
    shape = data_set.attrs['shape']
    return np.frombuffer(blosc.decompress(data_set), dtype=np.uint16).reshape(shape)
