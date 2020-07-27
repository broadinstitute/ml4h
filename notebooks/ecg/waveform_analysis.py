# %%
import pandas as pd
import numpy as np
import h5py
import logging
import seaborn as sns
from ml4cvd.TensorMap import decompress_data
from notebooks.ecg.waveform_plot import _ecg_rest_traces_and_text

mgh_df = pd.read_csv('/home/paolo/mgh_mrns_to_extract/c3po_mgh_qc_outcomes_05192020.csv')
mgh_df['acquisitiondatetime'] = pd.to_datetime(mgh_df['acquisitiondate'].apply(str)+ \
                                               'T' + \
                                               mgh_df['acquisitiontime'].apply(str))
mgh_df['distance_from_fu'] = pd.to_datetime(mgh_df['start_fu']) \
                             - pd.to_datetime(mgh_df['acquisitiondatetime'])
mgh_close_fu = mgh_df[['MRN', 'distance_from_fu']].groupby('MRN') \
                                                  .min() 
mgh_patients = mgh_df.merge(mgh_close_fu, on=['MRN', 'distance_from_fu'])

# %%
for i, row in mgh_patients[['patientid', 'acquisitiondate', 'acquisitiontime']].iterrows():
    try:
        mrn = int(float(row['patientid']))
        path = f"/data/ecg/mgh/{mrn}.hd5"
        with h5py.File(path, 'r') as hd5:
            hd5_out = h5py.File(f"/home/paolo/mgh_mrns_to_extract/mgh_3yrs_last_before_fu/{i}.hd5", 'w')
            hd5_out[f"partners_ecg_rest/{row['acquisitiondate']}T{row['acquisitiontime']}"] = h5py.ExternalLink(path, f"/partners_ecg_rest/{row['acquisitiondate']}T{row['acquisitiontime']}")
    except:
        pass

# %% 
from sklearn.model_selection import train_test_split
ids = list(range(len(mgh_patients)))
train, test = train_test_split(ids, test_size=0.10)
train, valid = train_test_split(train, test_size=0.30)
np.savetxt('/home/paolo/mgh_mrns_to_extract/mgh_3yrs_last_before_fu/train.csv',
           np.array(train, dtype=np.int), fmt="%i")
np.savetxt('/home/paolo/mgh_mrns_to_extract/mgh_3yrs_last_before_fu/valid.csv',
           np.array(valid, dtype=np.int), fmt="%i")
np.savetxt('/home/paolo/mgh_mrns_to_extract/mgh_3yrs_last_before_fu/test.csv',
           np.array(test, dtype=np.int), fmt="%i")


# %%
ff = h5py.File(f'/home/paolo/mgh_mrns_to_extract/mgh_3yrs_last_before_fu/3.hd5', 'r')
ff['partners_ecg_rest/2019-01-11T09:57:55'].keys()
from ml4cvd.arguments import _get_tmap
tm_input = _get_tmap('partners_ecg_2500_oldest', [])
arr = tm_input.tensor_from_file(tm_input, ff)
tm_output = _get_tmap('partners_ecg_bias_locationcardiology_oldest', [])
# tm_output = _get_tmap('partners_ecg_bias_locationname_oldest', [])
location = tm_output.tensor_from_file(tm_output, ff)
ff.close()

# %%
# ff = h5py.File(f'/data/partners_ecg/mgh/hd5/{int(mgh_patients.iloc[11]["patientid"])}.hd5', 'r')
# date = "T".join(str(mgh_patients.iloc[0]["acquisitiondatetime"]).split())
# decompress_data(ff[f'partners_ecg_rest/{date}/patientage'][()], ff[f'partners_ecg_rest/{date}/patientage'].attrs['dtype'])
# %%
# mgh_patients['age'] = '-1'

# for i, patient in mgh_patients.iterrows():
#     try:
#         with h5py.File(f'/data/partners_ecg/mgh/hd5/{int(patient["patientid"])}.hd5', 'r') as ff:
#             date = "T".join(str(patient["acquisitiondatetime"]).split())
#             try:
#                 age = decompress_data(ff[f'partners_ecg_rest/{date}/patientage'][()], ff[f'partners_ecg_rest/{date}/patientage'].attrs['dtype'])
#             except KeyError:
#                 age = '-1'
#             mgh_patients.loc[i, 'age'] = str(int(age))
#     except ValueError:
#         logging.info(f'Skipping patient {patient.patientid}')
# mgh_patients.to_csv('/home/paolo/mgh_mrns_to_extract/c3po_mgh_lastbeforefu_age_sex.csv', sep='\t', index=False)


# %%
%matplotlib inline
mgh_patients = pd.read_csv('/home/paolo/mgh_mrns_to_extract/c3po_mgh_lastbeforefu_age_sex.csv', sep='\t')
mgh_patients['acquisitionyear'] = mgh_patients['acquisitiondate'].str.split('-').str[0].apply(str)
sns.distplot(mgh_patients['age'])


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
nextract = 200
filter=True
patient_devices = {}
for device in bias_dic['acquisitiondevice']:
    patient_devices[device] = mgh_patients[(mgh_patients['acquisitiondevice']==device) & \
                                           (mgh_patients['I_len']==samples) & \
                                           (mgh_patients['I_nonzero']==10)]


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
    features[device]['firstbeat_raw'] = np.zeros((nextract, samples))
    features[device]['firstbeat_filter'] = np.zeros((nextract, samples))
    features[device]['spectrum'] = np.zeros((nextract, samples))
    features[device]['freqz'] = np.zeros((nextract, samples))
    features[device]['hp'] = np.zeros(nextract)
    features[device]['last'] = 0





# %%
from ml4cvd.tensor_maps_partners_ecg import _filter_voltage
for device in patient_devices:
    print(device)
    cnt = 0
    for (i, m) in patient_devices[device].iterrows():
        try:
            pid = int(m['patientid'])
        except:
            continue
        tensor_path = f'/data/partners_ecg/mgh/hd5/{pid}.hd5'
        tensor_date = 'T'.join(m['acquisitiondatetime'].split())
        with h5py.File(tensor_path, 'r') as hd5:
            leads, text = _ecg_rest_traces_and_text(hd5, tensor_date)
        features[device]['max'][cnt] = np.max(leads['I']['raw'])
        features[device]['min'][cnt] = np.max(leads['I']['raw'])
        try:
            features[device]['firstbeat'][cnt, :] = leads['I']['templates'][0:5].ravel()
        except:
            print('Skipping templates')
        if filter:
            leads['I']['raw'] = _filter_voltage(leads['I']['raw'])
        features[device]['firstbeat_raw'][cnt, :] = leads['I']['raw']
        features[device]['firstbeat_filtered'] = np.abs(np.fft.fft(leads['I']['filtered']))
        features[device]['spectrum'][cnt, :] = np.abs(np.fft.fft(leads['I']['raw']))
        features[device]['freqz'][cnt, :] = np.fft.fftfreq(samples, 10.0/samples)
        # features[device]['hp'][cnt]= hp
        features[device]['last'] = cnt
        cnt += 1
        if cnt == nextract : break

# %%
sub_devices = ['MAC55', 'MAC', 'MAC5K']
colors = [[0.0, 0.0, 0.0],
          [0.4, 0.4, 0.4],
          [0.8, 0.8, 0.8]
          ]
f, ax = plt.subplots()
freq = np.fft.fftfreq(samples, 10.0/samples)
for i, device in enumerate(sub_devices):
    ax.plot(np.fft.fftshift(np.median(features[device]['freqz'][:features[device]['last']], axis=0)), 
            np.fft.fftshift(np.median(features[device]['spectrum'][:features[device]['last']], axis=0)),
            color=colors[i], label=f'{device} n={features[device]["last"]}')
    #ax.set_xlim([50.0, 70.0])
    #ax.set_ylim([0.0, 2000.0])
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Magnitude spectrum')
    ax.legend()
f.savefig(f'/home/paolo/mgh_mrns_to_extract/waveform_analysis/device_frequency_notch_{samples}.png', dpi=500)

# %%

f, ax = plt.subplots()
for i, device in enumerate(sub_devices):
    ax.plot(np.median(features[device]['firstbeat'][:features[device]['last']], axis=0), 
            label=f'{device} n={features[device]["last"]}', color=colors[i])
ax.legend()
ax.set_xlabel('Beat number')
ax.set_xticks([50, 200, 350, 500, 650])
ax.set_xticklabels(['1', '2', '3', '4', '5'])
ax.set_ylabel('Amplitude (uV)')
f.savefig(f'/home/paolo/mgh_mrns_to_extract/waveform_analysis/device_time_{samples}.png', dpi=500)

f, ax = plt.subplots()
for device in sub_devices:
    ax.plot(np.median(features[device]['firstbeat_raw'][:features[device]['last']], axis=0), 
                      label=f'{device} n={features[device]["last"]}', color=colors[i])
ax.legend()

# %%
mgh_select_bias_keys = ['acquisitionyear', 'acquisitionsoftwareversion', 'acquisitiondevice', 
                        'testreason', 'I_nonzero', 'roomid', 'I_len', 'age', 'sex']
mgh_select_bias_keys = ['age', 'sex']

# %%
import statsmodels.api as sm

n_locationcardiology = np.sum(mgh_patients_numbers['locationcardiology']==1)
mgh_test = mgh_patients_numbers[mgh_patients['locationcardiology']==1]
models = []
ress = []
for i in range(10):
    mgh_control = mgh_patients_numbers[mgh_patients['locationcardiology']==0].sample(n_locationcardiology)
    mgh_covariates = pd.concat([mgh_test, mgh_control])[mgh_select_bias_keys]
    mgh_covariates['intercept'] = 1.0
    models.append(sm.Logit(pd.concat([mgh_test, mgh_control])[['locationcardiology']], mgh_covariates))
    ress.append(models[-1].fit())

# %%
from matplotlib.ticker import PercentFormatter

mgh_patients_numbers['scores'] = 0.0
mgh_covariates = mgh_patients_numbers[mgh_select_bias_keys]
mgh_covariates['intercept'] = 1.0
for model, res in zip(models, ress):
    scores = model.predict(res.params, mgh_covariates, linear=False)
    mgh_patients_numbers.loc[:, 'scores'] += scores
mgh_patients_numbers.loc[:, 'scores'] /= len(ress)

f, ax = plt.subplots()
controls = mgh_patients_numbers[mgh_patients_numbers['locationcardiology']==0]
tests = mgh_patients_numbers[mgh_patients_numbers['locationcardiology']==1]
ax.hist(controls['scores'], weights=np.ones(len(controls))/len(controls),
        alpha=0.5, bins=30, label='non-cardiology location')
ax.hist(tests['scores'], weights=np.ones(len(tests))/len(tests),
        alpha=0.5, bins=30, label='cardiology location')   
ax.yaxis.set_major_formatter(PercentFormatter(xmax=1))
ax.set_xlabel('Propensity scores')
plt.legend()
f.savefig('/home/paolo/mgh_mrns_to_extract/propensity_scores_age_sex.png', dpi=500)


# %%

mgh_patients_tests = mgh_patients_numbers[mgh_patients_numbers['locationcardiology']==1]
mgh_patients_tests['matched'] = 0
mgh_patients_tests['scores_diff'] = 1000.0
mgh_patients_controls = mgh_patients_numbers[mgh_patients_numbers['locationcardiology']==0]
for i, patient in mgh_patients_tests.iterrows():
    tmp = mgh_patients_controls.iloc[(mgh_patients_controls['scores']-patient['scores']).abs().argsort()[:100]]
    tmp = tmp.sample(1)
    mgh_patients_tests.loc[i, 'matched'] = tmp['patientid'].values
    mgh_patients_tests.loc[i, 'scores_diff'] = abs(tmp['scores'].values - patient['scores'])


# %%
mgh_patients_controls_matched = pd.DataFrame()
mgh_patients_controls_matched['patientid'] = mgh_patients_tests['matched']
mgh_patients_controls_matched = mgh_patients_controls_matched.merge(mgh_patients_controls, on='patientid')
mgh_patients_controls_matched['locationcardiology']

# %%
mgh_patients_matched = pd.concat([mgh_patients_tests, mgh_patients_controls_matched])

# %%
import scipy.stats
def proportion_test_controls(df, tests, controls):
    tab = pd.crosstab(df[tests], df[controls], margins=True)
    vals = []
    xlabels = []
    for key in mgh_bias_dic[controls]:
        xlabels.append(key)
        val = mgh_bias_dic[controls][key]
        if val not in tab.keys():
            vals.append(0)
            continue
        prop_controls = tab.loc[0, val] / tab.loc[0, 'All']
        prop_tests = tab.loc[1, val] / tab.loc[1, 'All']
        vals.append(prop_tests - prop_controls)
    return xlabels, vals

for key in mgh_select_bias_keys:
    xlabels, vals_unmatched = proportion_test_controls(mgh_patients_numbers, 'locationcardiology', key)
    xlabels, vals_matched = proportion_test_controls(mgh_patients_matched, 'locationcardiology', key)
    x = np.arange(len(vals_matched))
    f, ax = plt.subplots()
    ax.bar(x, vals_unmatched, width=0.5, color='black', label='before matching')
    ax.bar(x+0.5, vals_matched, width=0.5, color='white', edgecolor='black', label='after matching')
    chi2_unmatched = scipy.stats.chi2_contingency(pd.crosstab(mgh_patients_numbers['locationcardiology'],
                                                              mgh_patients_numbers[key]))
    chi2_matched = scipy.stats.chi2_contingency(pd.crosstab(mgh_patients_matched['locationcardiology'],
                                                              mgh_patients_matched[key]))
    ax.set_xlabel(key)
    ax.set_ylabel('Proportional difference (tests-control)')
    ax.set_xticks(x)
    ax.set_xticklabels(xlabels)
    plt.legend()
    ax.set_title(f'p-value unmatched|matched: {chi2_unmatched[1]:.2f}|{chi2_matched[1]:.2f}')
    f.savefig(f'/home/paolo/mgh_mrns_to_extract/propensity_matched_age_sex_{key}.png', dpi=500)
# %%
mgh_patients_matched['patientid']

# %%
mgh_patients_matched = mgh_patients_matched[['patientid']].merge(mgh_patients, on = 'patientid')
#mgh_patients_matched['patientid'] = mgh_patients_matched['patientid'].apply(int)
#%%
for i, row in mgh_patients_matched[['patientid', 'acquisitiondate', 'acquisitiontime']].iterrows():
    #try:
        mrn = int(float(row['patientid']))
        path = f"/data/partners_ecg/mgh/hd5/{mrn}.hd5"
        with h5py.File(path, 'r') as hd5:
            hd5_out = h5py.File(f"/home/paolo/mgh_mrns_to_extract/mgh_3yrs_last_before_fu_matched_age_sex/{i}.hd5", 'w')
            hd5_out[f"partners_ecg_rest/{row['acquisitiondate']}T{row['acquisitiontime']}"] = h5py.ExternalLink(path, f"/partners_ecg_rest/{row['acquisitiondate']}T{row['acquisitiontime']}")
    ##except:
    #    print(f"Failed to {row['patientid']}")

# %%
from sklearn.model_selection import train_test_split
ids = list(range(len(mgh_patients_matched)))
train, test = train_test_split(ids, test_size=0.10)
train, valid = train_test_split(train, test_size=0.30)
np.savetxt('/home/paolo/mgh_mrns_to_extract/mgh_3yrs_last_before_fu_matched_age_sex/train.csv',
           np.array(train, dtype=np.int), fmt="%i")
np.savetxt('/home/paolo/mgh_mrns_to_extract/mgh_3yrs_last_before_fu_matched_age_sex/valid.csv',
           np.array(valid, dtype=np.int), fmt="%i")
np.savetxt('/home/paolo/mgh_mrns_to_extract/mgh_3yrs_last_before_fu_matched_age_sex/test.csv',
           np.array(test, dtype=np.int), fmt="%i")

# %%
f, ax = plt.subplots()
f.set_size_inches(16, 9)
mgh_patients_matched.hist(ax=ax)
plt.tight_layout()
f.savefig('/home/paolo/mgh_mrns_to_extract/diagnose_matching.png', dpi=500)

# %%
plt.hist(mgh_patients_matched['MRN'], bins=1000)

# %%
