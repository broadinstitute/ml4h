# %%
import pandas as pd
import numpy as np
import seaborn as sns
import glob
import h5py
from ml4cvd.tensor_maps_partners_ecg import TMAPS
from scipy.signal import butter, lfilter

# %%
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='bandpass')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def butter_lowpass(cutOff, fs, order=5):
    nyq = 0.5 * fs
    normalCutoff = cutOff / nyq
    b, a = butter(order, normalCutoff, btype='low')
    return b, a

def butter_lowpass_filter(data, cutOff, fs, order=4):
    b, a = butter_lowpass(cutOff, fs, order=order)
    y = lfilter(b, a, data)
    return y


# %%
hd5s = glob.glob('/home/paolo/pdfs/*.hd5')

#%%
import matplotlib.pyplot as plt
%matplotlib inline
tm_pdf = TMAPS['partners_ecg_1250_pdf'] 
tm = TMAPS['partners_ecg_5000_raw']
tm_datetime = TMAPS['partners_ecg_datetime']
tm_sample_freq = TMAPS['partners_ecg_sampling_frequency_wv']
ratios = {}
cnt = 0
for i, hd5 in enumerate(hd5s):    
    with h5py.File(hd5) as tensor:
        arr_pdf = tm_pdf.tensor_from_file(tm_pdf, tensor)
        arr = tm.tensor_from_file(tm, tensor)
        for j in range(arr_pdf.shape[0]):
            if np.sum(np.abs(arr_pdf[j][:1250, 0])) < 1e-3:
                continue
            ratios[f'{hd5}_{j}'] = {}
            ratios[f'{hd5}_{j}']['pdf'] = arr_pdf[j][:1250, 0]
            ratios[f'{hd5}_{j}']['xml'] = arr[j][:1250, 0]
            ratios[f'{hd5}_{j}']['ratio'] = np.abs(np.fft.fft(arr_pdf[j][:1250, 0])/np.fft.fft(arr[j][:1250, 0]))
        # arr = butter_lowpass_filter(arr, 40.0, 240.0, order=6)
    #     dates = tm_datetime.tensor_from_file(tm_datetime, tensor)
    #     samples = tm_sample_freq.tensor_from_file(tm_sample_freq, tensor)
            mrn = hd5.split('/')[-1].replace('.hd5', '')
            f, ax = plt.subplots()
            dt_pdf = 2.5 / 1250.0
            time_pdf = np.arange(0, 1250*dt_pdf, dt_pdf)
            dt = 1.0/247.0/2.0
            time = np.arange(0, len(ratios[f'{hd5}_{j}']['xml'])*dt, dt)
            ax.plot(time, ratios[f'{hd5}_{j}']['xml'], linewidth=3, color='black', label='XML')
            ax.plot(time_pdf, ratios[f'{hd5}_{j}']['pdf']+100.0, linewidth=3, color='gray', label='PDF')
            ax.set_ylabel('lead I voltage ($\mu$V)')
            ax.set_xlabel('Time (s)')
            ax.set_xlim([0, 2.5])
            ax.set_ylim([-1000, 2000])
            plt.legend()
            x = np.fft.fftfreq(1250, 2.5/1250.0)
            y = np.abs(np.fft.fft(ratios[f'{hd5}_{j}']['xml']/len(ratios[f'{hd5}_{j}']['xml'])))
            y_pdf = np.abs(np.fft.fft(ratios[f'{hd5}_{j}']['pdf'])/1250.0)
            # f.savefig(f'/home/paolo/{mrn}_time.png', dpi=500)
            f, ax = plt.subplots(3, 1)
            ax[0].plot(x, y, linewidth=3, color='black', label='XML')
            ax[1].plot(x, y_pdf, linewidth=3, color='gray', label='PDF')
            ax[2].plot(x, y_pdf/y, linewidth=3, color='gray', label='ratio')
            ax[0].set_ylim([0, 50])
            ax[1].set_ylim([0, 50])
            ax[0].set_title(mrn)
            ax[0].legend()
            ax[1].legend()
            cnt += 1
        if cnt >= 50: 
            break

        # f.savefig(f'/home/paolo/{mrn}_freq.png', dpi=500)

# %%
mean = np.zeros((len(ratios), 1250))
for i, hd5 in enumerate(ratios):
    mean[i] = ratios[hd5]['ratio']
mean_arr = np.median(mean, axis=0)

f, ax = plt.subplots()
ax.plot(np.fft.fftfreq(1250, 2.5/1250), mean_arr)
# ax.set_ylim([0.0, 4.0])
ax.set_xlim([-200.0, 200.0])

# %%

mgh_patients = pd.read_csv('/home/paolo/mgh_mrns_to_extract/c3po_mgh_lastbeforefu_age_sex.csv', sep='\t', dtype=str)
mgh_patients['patientid'] = pd.to_numeric(mgh_patients['patientid'], errors='coerce', downcast='integer')
mgh_patients['patientid'] = mgh_patients['patientid'].apply(lambda x: f'{x:09.0f}')
mgh_patients['acquisitionyear'] = mgh_patients['acquisitiondate'].str.split('-').str[0].apply(str)
#sns.distplot(mgh_patients['age'])

mgh_patients['patientid'].to_csv('/home/paolo/pdfs/mrns.csv', index=False, header=False, float_format='%09d')

# %%
import matplotlib.pyplot as plt
age = mgh_patients['age'].apply(int)
prev_decade = -10
for decade in range(30, 100, 10):
    print(decade)
    mgh_patients.loc[(age<decade) & (age>=prev_decade), 'age'] = str(decade-10)
    prev_decade = decade
mgh_patients.loc[age>=80] = '80'
mgh_patients['age'].describe()
plt.hist(mgh_patients['age'].apply(int), bins=7)

# %%
# %%
bias_dic = {'acquisitiondevice': {'MAC': 0, 'MAC55': 1, 'MAC5K': 2, 'D3K': 3, 'MACVU': 4, 'S8500': 5, 'CASE': 6, 'MAC16': 7, 'MAC 8': 8, 'unspecified': 9},
'acquisitionsoftwareversion': {'nan': 0, '010A': 1, '009A': 2, '007A.2': 3, '005A.1': 4, '006A': 5, '009C': 6, '010B': 7, '008A': 8, '008B': 9, 'unspecified': 10},
'analysissoftwareversion': {'22': 0, '14': 1, '231': 2, '241 HD': 3, '239': 4, '26': 5, '237': 6, '235': 7, '233': 8, '241': 9, 'unspecified': 10}, 
'cartnumber': {'1.0': 0, '0.0': 1, '2.0': 2, '14.0': 3, '3.0': 4, '4.0': 5, '101.0': 6, '127.0': 7, '102.0': 8, '126.0': 9, 'unspecified': 10}, 
'locationname': {'46-YAWKEY5 - CARDIOLOGY NR': 0, '30-EMERGENCY DEPARTMENT': 1, '40-WACC2/6 - CLINICS': 2, '160-LUNDER EMERGENCY DEPARTMENT': 3, '23-JACKSON 121-SURGICAL DAY CARE': 4, '7-ELLISON 10 - CARDIAC': 5, '53-WACC 5 BUL MED GROUP': 6, '22-PRIVATE AMBULATORY': 7, '44-PROCESS DO NOT INTERPRET': 8, '106-BIGELOW8-CARDIO SUITE 800 NR': 9, 'unspecified': 10}, 'overreaderid': {'999.0': 0, '888.0': 1, '3.0': 2, '32.0': 3, '80.0': 4, '15.0': 5, '103.0': 6, '57.0': 7, '18.0': 8, '131.0': 9, 'unspecified': 10}, 
'priority': {'NORMAL': 0, 'PREOP': 1, 'STAT': 2, 'unspecified': 3}, 
'roomid': {'nan': 0, '99': 1, 'BAY3': 2, '57': 3, 'BAY4': 4, 'BAY1': 5, 'BAY6': 6, 'BAY9': 7, 'BAY2': 8, 'BAY7': 9, 'unspecified': 10}, 
'testreason': {'nan': 0, 'V72.81': 1, '786.50': 2, 'NOBILL': 3, '401.9': 4, '00': 5, '785.1': 6, '57': 7, '786.09': 8, 'V71.7': 9, 'unspecified': 10}, 
'I_len': {'2500.0': 0, '5000.0': 1, 'unspecified': 2}, 
'I_nonzero': {'10.0': 0, '5.0': 1, '0.0': 2, '2.5': 3, 'unspecified': 4}, 
'II_nonzero': {'10.0': 0, '5.0': 1, '0.0': 2, '2.5': 3, 'unspecified': 4}, 
'III_nonzero': {'10.0': 0, '5.0': 1, '2.5': 2, '0.0': 3, 'unspecified': 4}, 
'V1_nonzero': {'10.0': 0, '2.5': 1, '0.0': 2, '5.0': 3, 'unspecified': 4}, 
'V2_nonzero': {'10.0': 0, '2.5': 1, '0.0': 2, '5.0': 3, 'unspecified': 4}, 
'V3_nonzero': {'10.0': 0, '2.5': 1, '0.0': 2, '5.0': 3, 'unspecified': 4}, 
'V4_nonzero': {'10.0': 0, '2.5': 1, '0.0': 2, '5.0': 3, 'unspecified': 4}, 
'V5_nonzero': {'10.0': 0, '2.5': 1, '0.0': 2, '5.0': 3, 'unspecified': 4}, 
'V6_nonzero': {'10.0': 0, '2.5': 1, '0.0': 2, '5.0': 3, 'unspecified': 4}, 
'aVR_nonzero': {'10.0': 0, '5.0': 1, '2.5': 2, '0.0': 3, 'unspecified': 4}, 
'aVL_nonzero': {'10.0': 0, '5.0': 1, '2.5': 2, '0.0': 3, 'unspecified': 4}, 
'aVF_nonzero': {'10.0': 0, '5.0': 1, '2.5': 2, '0.0': 3, 'unspecified': 4},
'sex': {'F': 0, 'M': 1, 'unspecified': 2}, 
'race': {'WHITE': 0, 'BLACK OR AFRICAN AMERICAN': 1, 'OTHER': 2, 'ASIAN': 3, 'OTHER@HISPANIC': 4, 'UNKNOWN': 5, 'DECLINED': 6, 'HISPANIC OR LATINO': 7, 'HISPANIC': 8, 'BLACK': 9, 'unspecified': 10}, 
           }

bias_dic['acquisitionyear'] = {str(year): i for i, year in enumerate(range(1998, 2020))}
bias_dic['age'] = {str(age): i for i, age in enumerate(range(20, 89, 10))}

# %%
mgh_bias_dic = {}
for col in bias_dic:
    mgh_bias_dic[col] = {}
    gb = mgh_patients[[col, 'patientid']] \
                        .groupby([col])['patientid'] \
                        .count() \
                        .reset_index(name='count') \
                        .sort_values('count', ascending=False) \
                        .head(30)
    mgh_bias_dic[col] = {str(val): i for i, val in enumerate(gb[col])}
    mgh_bias_dic[col]['unspecified'] = len(gb[col])

# %%
locationcardiology = {
                             '46-YAWKEY5 - CARDIOLOGY NR': 1,
                             '30-EMERGENCY DEPARTMENT': 0,
                             '40-WACC2/6 - CLINICS': 0,
                             '160-LUNDER EMERGENCY DEPARTMENT': 0,
                             '23-JACKSON 121-SURGICAL DAY CARE': 0,
                             '7-ELLISON 10 - CARDIAC': 1,
                             '53-WACC 5 BUL MED GROUP': 0,
                             '22-PRIVATE AMBULATORY': 0,
                             '44-PROCESS DO NOT INTERPRET': 0,
                             '106-BIGELOW8-CARDIO SUITE 800 NR': 1,
                             '18-GREY 1 ADMITTING TEST AREA': 0,
                             '43-CHELSEA HLTH CNTR LAB': 0,
                             '6-ELLISON 9 - CCU': 1,
                             '8-ELLISON 11 - CARDIAC': 1,
                             '2-BLAKE 8 - CARDIAC SICU': 1,
                             '91-REVERE HLTH CNTR LAB': 0,
                             '89-WHITE 9 - MED': 0,
                             '33-ELLISON 16 - MED': 0,
                             '71-BIGELOW 11 - MED': 0,
                             '88-WHITE 8 - MED': 0,
                             '98-BUNKER HILL HEALTH CENTER NR': 0,
                             '63-MGH BACK BAY': 0,
                             '59-CHELSEA HEALTH CENTER ED NR': 0,
                             '80-WHITE 10 - MED': 0,
                             '81-WHITE 11 - UROLOGY': 0,
                             '5-ELLISON 8 - CARDIAC SURG': 1,
                             '110-ED TRAUMA': 0,
                             '101-BEACON HILL PRIMARY NR': 0,
                             '138-ED OBSERVATION UNIT': 0,
                             '147-DANVERS ACC CARDIOLOGY PRTCE': 1,
                             'unspecified': 0}


def location_is_cardiology(location):
    for key in locationcardiology:
        if str(location).lower() == key.lower():
            return locationcardiology[key]
    return 0

mgh_patients['locationcardiology'] = mgh_patients['locationname'].apply(location_is_cardiology)

# %%
# %%
def dec_convert(synonyms, unspecified_key='unspecified'):
    def convert_key_to_class(value):
        for key in synonyms:
            if str(value).lower() == key.lower():
                return synonyms[key]
        return synonyms[unspecified_key]
    return convert_key_to_class

mgh_patients_numbers = pd.DataFrame()
for key in mgh_bias_dic:
    mgh_patients_numbers[key] = mgh_patients[key].apply(dec_convert(mgh_bias_dic[key]))
mgh_patients_numbers['locationcardiology'] = mgh_patients['locationcardiology']
mgh_patients_numbers['patientid'] = mgh_patients['patientid']

# %%
#from notebooks.ecg.waveform_plot import plot_ecg_rest
samples = 2500
nextract = 3
filter=False
mgh_patients['I_len'] = mgh_patients['I_len'].apply(float)
mgh_patients['I_nonzero'] = mgh_patients['I_nonzero'].apply(float)
patient_devices = {}
for device in bias_dic['acquisitiondevice']:
    patient_devices[device] = mgh_patients[(mgh_patients['acquisitiondevice']==device) & \
                                           (mgh_patients['I_len']<samples+1) & \
                                           (mgh_patients['I_len']>samples-1) & \
                                           (mgh_patients['I_nonzero']<10.1) & \
                                           (mgh_patients['I_nonzero']>9.9)]


# # %%
# import pathlib
# pathlib.Path(f"/home/paolo/mgh_mrns_to_extract/mgh_3yrs_last_before_{samples}/").mkdir(parents=True, exist_ok=True)
# mgh_patients_samples = mgh_patients[(mgh_patients['I_len']==samples) & \
#                                     (mgh_patients['I_nonzero']==10)]
# for i, row in mgh_patients_samples.iterrows():
#     try:
#         mrn = int(float(row['patientid']))
#     except ValueError:
#         pass
#     path = f"/data/partners_ecg/mgh/hd5/{mrn}.hd5"
#     with h5py.File(path, 'r') as hd5:
#         hd5_out = h5py.File(f"/home/paolo/mgh_mrns_to_extract/mgh_3yrs_last_before_{samples}/{i}.hd5", 'w')
#         hd5_out[f"partners_ecg_rest/{row['acquisitiondate']}T{row['acquisitiontime']}"] = h5py.ExternalLink(path, f"/partners_ecg_rest/{row['acquisitiondate']}T{row['acquisitiontime']}")

# # %%
# from sklearn.model_selection import train_test_split
# ids = list(range(len(mgh_patients_samples)))
# train, test = train_test_split(ids, test_size=0.10)
# train, valid = train_test_split(train, test_size=0.30)
# np.savetxt(f'/home/paolo/mgh_mrns_to_extract/mgh_3yrs_last_before_{samples}/train.csv',
#            np.array(train, dtype=np.int), fmt="%i")
# np.savetxt(f'/home/paolo/mgh_mrns_to_extract/mgh_3yrs_last_before_{samples}/valid.csv',
#            np.array(valid, dtype=np.int), fmt="%i")
# np.savetxt(f'/home/paolo/mgh_mrns_to_extract/mgh_3yrs_last_before_{samples}/test.csv',
#            np.array(test, dtype=np.int), fmt="%i")


# %%
features = {}
for device in patient_devices:
    features[device] = {}
    features[device]['max'] = np.zeros(nextract,)
    features[device]['min'] = np.zeros(nextract,)
    features[device]['firstbeat'] = np.zeros((nextract, 750))
    features[device]['firstbeat_raw'] = np.zeros((nextract, 625))
    features[device]['firstbeat_pdf'] = np.zeros((nextract, 625))
    features[device]['firstbeat_filter'] = np.zeros((nextract, 625))
    features[device]['spectrum'] = np.zeros((nextract, 625))
    features[device]['spectrum_pdf'] = np.zeros((nextract, 625))
    features[device]['spectrum_ratio'] = np.zeros((nextract, 625))
    features[device]['freqz'] = np.zeros((nextract, 625))
    features[device]['freqz_pdf'] = np.zeros((nextract, 625))
    features[device]['hp'] = np.zeros(nextract)
    features[device]['last'] = 0

# %%
from ast import literal_eval
pdf_path = f'/home/paolo/pdfs_compare/{patient_devices["MAC5K"].iloc[0].patientid}.csv'
pdf = pd.read_csv(pdf_path)
def string_to_array(string):
    string = string.replace('\n', '')
    string = ','.join(string.strip().split())
    string = string[0] + string[2:]
    array = literal_eval(string)
    return np.array(array)
pdf['clean'] = pdf['signal'].apply(string_to_array)

# %%
import h5py
from ml4cvd.tensor_maps_partners_ecg import _filter_voltage
# from ml4cvd.notebooks.ecg.waveform_plot import _ecg_rest_traces_and_text
from scipy.interpolate import interp1d

for device in ['MAC5K']:
    print(device)
    cnt = 0
    for (i, m) in patient_devices[device].iterrows():
        try:
            pid = int(m['patientid'])
        except:
            continue
        tensor_path = f'/home/paolo/mgh_mrns_to_extract/mgh_3yrs_last_before_fu/{pid}.hd5'
        pdf_path = f'/home/paolo/pdfs_compare/{m["patientid"]}.csv'
        pdf = pd.read_csv(pdf_path)
        pdf['clean'] = pdf.signal.apply(string_to_array)
        lead_I = pdf[pdf['lead']=='I'].clean.values[0]
        lead_I_int = interp1d(np.linspace(0.0, 624, len(lead_I)), lead_I)
        lead_I = lead_I_int(np.arange(625))
        tensor_date = 'T'.join(m['acquisitiondatetime'].split())
        with h5py.File(tensor_path, 'r') as hd5:
            arr = tm.tensor_from_file(tm, hd5)
        features[device]['max'][cnt] = np.max(arr[0][:625, 0])
        features[device]['min'][cnt] = np.min(arr[0][:625, 0])
        try:
            features[device]['firstbeat'][cnt, :] = leads['I']['templates'][0:5].ravel()
        except:
            print('Skipping templates')
        features[device]['firstbeat_raw'][cnt, :] = arr[0][:625, 0]
        features[device]['firstbeat_pdf'][cnt, :len(lead_I)] = lead_I
        features[device]['spectrum'][cnt, :] = np.abs(np.fft.fft(arr[0][:625, 0]))/625
        features[device]['freqz'][cnt, :] = np.fft.fftfreq(625, 2.5/625)
        features[device]['spectrum_pdf'][cnt, :] = np.abs(np.fft.fft(lead_I))/625
        features[device]['spectrum_ratio'][cnt, :] = np.abs(np.fft.fft(lead_I)/np.fft.fft(arr[0][:625, 0]))
        features[device]['freqz_pdf'][cnt, :] = np.fft.fftfreq(625, 2.5/625)
        # features[device]['hp'][cnt]= hp
        features[device]['last'] = cnt
        cnt += 1
        if cnt == nextract : break

# %%
%matplotlib inline
cnt = 0
for (i, m) in patient_devices[device].iterrows():
    print(m.acquisitiondevice)
    if cnt == nextract : break
    f, ax = plt.subplots()
    f.set_size_inches(4, 3)
    ax.set_title(m['patientid'])
    ax.plot(features[device]['firstbeat_raw'][cnt, :], color='black', label='XML')
    ax.plot(features[device]['firstbeat_pdf'][cnt, :625], color=[0.6, 0.6, 0.6], label='PDF')
    ax.set_xlabel('# sample')
    ax.set_ylabel('lead I voltage ($\mu$V)')
    plt.legend()
    plt.tight_layout()
    f.savefig(f'time_domain_{m["patientid"]}.png', dpi=500)
    f, ax = plt.subplots(2, 1)
    f.set_size_inches(4, 3)
    ax[0].plot(features[device]['freqz_pdf'][cnt, :],
               features[device]['spectrum'][cnt, :], color='black', label='XML')   
    ax[0].set_xticklabels([])
    ax[0].legend()
    ax[1].plot(features[device]['freqz_pdf'][cnt, :],
               features[device]['spectrum_ratio'][cnt, :],
               color=[0.6, 0.6, 0.6], label='PDF')
    ax[1].legend()
    ax[1].set_xlabel('Frequency (Hz)')
    f.savefig(f'freq_domain_{m["patientid"]}.png', dpi=500)
    cnt += 1

# %%


# %%

# %%
