# %% 
import bs4
import h5py
import numpy as np
from collections import defaultdict
from ml4cvd.tensor_writer_partners import _data_from_xml
from typing import Dict, Union
from ml4cvd.defines import ECG_REST_AMP_LEADS
from ml4cvd.tensor_writer_partners import _compress_and_save_data
ECG_REST_AMPLITUDE_LEADS = {"I": 0, "II": 1, "III": 2, "AVR": 3, "AVL": 4, "AVF": 5, 
                            "V1": 6, "V2": 7, "V3": 8, "V4": 9, "V5": 10, "V6": 11}

# %%
ecg_data = dict()

# define tags that we want to find and use SoupStrainer to speed up search    
tags = [
        'amplitudemeasurements',
    ]
strainer = bs4.SoupStrainer(tags)

# %%
def _get_amplitude_from_amplitude_tags(amplitude_tags: bs4.ResultSet) -> Dict[str, Union[str, Dict[str, np.ndarray]]]:
    amplitude_data = {}
    amplitude_features = ['peak', 'start', 'duration', 'area']
    wave_ids = set()
    for amplitude_tag in amplitude_tags:
        lead_id = amplitude_tag.find('amplitudemeasurementleadid').text
        wave_id = amplitude_tag.find('amplitudemeasurementwaveid').text
        if wave_id not in wave_ids:
            wave_ids.add(wave_id)
            for amplitude_feature in amplitude_features:
                amplitude_data[f'{wave_id}_{amplitude_feature}'] = np.empty(len(ECG_REST_AMPLITUDE_LEADS))
                amplitude_data[f'{wave_id}_{amplitude_feature}'][:] = np.nan
        for amplitude_feature in amplitude_features:
            value = int(amplitude_tag.find(f'amplitudemeasurement{amplitude_feature}').text)
            amplitude_data[f'{wave_id}_{amplitude_feature}'][ECG_REST_AMPLITUDE_LEADS[lead_id]] = value
    return amplitude_data


# %%
fpath_xml = '/data/partners_ecg/mgh/xml_not_tensorized/mgh-vm1/MUSE_20200427_023949_15000.xml'
with open(fpath_xml, 'r') as f:
    soup = bs4.BeautifulSoup(f, 'lxml', parse_only=strainer)

# %%
ecg_data['amplitude'] = _get_amplitude_from_amplitude_tags(soup.find_all('measuredamplitude'))
with h5py.File('/home/paolo/test.hd5', 'w') as ff:
    gp = ff.create_group('test')
    amplitude = ecg_data.pop('amplitude')
    for wave_feature in amplitude:
        _compress_and_save_data(gp, wave_feature, amplitude[wave_feature], dtype='float')


# %%
from ml4cvd.tensor_writer_partners import _data_from_xml
ll = _data_from_xml('/data/partners_ecg/mgh/xml_not_tensorized/mgh-vm1/MUSE_20200427_023949_15000.xml')

# %%
ll['amplitude']

# %%
from ml4cvd.tensor_writer_partners import _convert_xml_to_hd5
import h5py
with h5py.File('/home/paolo/test2.hd5', 'a') as hd5:
    hd5_ecg = hd5.create_group('blabla')
    _convert_xml_to_hd5('/data/partners_ecg/mgh/xml_not_tensorized/mgh-vm1/MUSE_20200427_023949_15000.xml',
                        '/home/paolo/test2.hd5', hd5_ecg)

# %%
from ml4cvd.tensor_writer_partners import write_tensors_partners

write_tensors_partners('/data/partners_ecg/mgh/xml_not_tensorized/mgh-vm1/', 
                       '/home/paolo/test/', 1)


# %%
from ml4cvd.TensorMap import decompress_data
import h5py
ff = h5py.File('/home/paolo/test/1235562.hd5')
for date in ff['partners_ecg_rest']:
    print(decompress_data(ff['partners_ecg_rest'][date]['measuredamplitudepeak_IE_R'][()],
    dtype=ff['partners_ecg_rest'][date]['measuredamplitudepeak_IE_R'].attrs['dtype']))

# %%
