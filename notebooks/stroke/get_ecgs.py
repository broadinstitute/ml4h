# %%
import shutil
import pandas as pd
import os
import h5py
import numpy as np
registry = pd.read_csv('stroke_registry_mrn_mapped_toast_subtype-062020.txt', sep='\t')
index_date = pd.read_csv('index_date.csv', sep=',')
wide = pd.read_csv('wide_06-29-20.tsv', sep='\t')

registry = registry.merge(index_date, on='MRN')
registry = registry.merge(wide[['id', 'BMI', 'Systolic_BP', 'Diastolic_BP']], left_on='linker_id', right_on='id')
registry['BMI'] = registry['BMI'].fillna(20.0)
registry['Systolic_BP'] = registry['Systolic_BP'].fillna(120.0)
registry['Diastolic_BP'] = registry['Diastolic_BP'].fillna(80.0)
registry['index_date'] = pd.to_datetime(registry['Date Index Event'], errors='coerce')
registry = registry[(registry['Toast'] < 4.5) | (registry['Toast'] > 5.5)]
registry.loc[registry['Toast']>5.5, 'Toast'] = 5
# %%
cnt = 0
registry['ecg_date'] = 0.0
registry['n_ecgs'] = 0.0
for i, patient in registry.iterrows():
    if os.path.isfile(f'/data/partners_ecg/mgh/{int(patient["MRN"])}.hd5'):
        with h5py.File(f'/home/paolo/stroke_ecgs/{int(patient["MRN"])}.hd5', 'w') as hd5:
            hd5['partners_ecg_rest'] = h5py.ExternalLink(f'/data/partners_ecg/mgh/{int(patient["MRN"])}.hd5', '/partners_ecg_rest')
            ecg_dates = pd.to_datetime(list(hd5['partners_ecg_rest']))
            registry.loc[registry['MRN']==patient['MRN'], 'n_ecgs'] = len(hd5['partners_ecg_rest'])
            close_event = np.argmin(np.abs(ecg_dates-patient['index_date']))
            ecg_date = ecg_dates[close_event]
            registry.loc[registry['MRN']==patient['MRN'], 'ecg_date'] = ecg_date
            link_file = h5py.File(f'/home/paolo/stroke_ecgs_one_per_patient/{int(patient["MRN"])}.hd5', 'w')
            link_file[f'partners_ecg_rest/{"T".join(str(ecg_date).split())}'] = h5py.ExternalLink(f'/data/partners_ecg/mgh/{int(patient["MRN"])}.hd5', f'partners_ecg_rest/{"T".join(str(ecg_date).split())}')
            link_file.close()

registry['ecg_date'] = pd.to_datetime(registry['ecg_date'], errors='coerce')
registry = registry.dropna()

# %%
from matplotlib import pyplot as plt
import matplotlib
%matplotlib inline
f, ax = plt.subplots()
ax.hist(registry[registry['n_ecgs']<100]['n_ecgs'], bins=100, edgecolor='black', color='white')
ax.set_xlabel('# ECGs available per patient')
ax.set_ylabel('# patients')
ax.set_xlim([1, 100])
medn = np.median(registry[registry['n_ecgs']<100]['n_ecgs'])
ax.plot([medn,medn], [0, 300], 'k--')
ax.set_ylim([0, 300])
xticks = ax.get_xticklabels()
xticks = [xticks[0]]+[matplotlib.text.Text(medn, 0, str(medn))]+xticks[1:]
ax.set_xticks([0.0, medn, 20.0, 40.0, 60.0, 80.0, 100.0])
f.savefig('n_ecgs_per_patient.png', dpi=500)

# %%
from matplotlib import pyplot as plt
import matplotlib
%matplotlib inline
f, ax = plt.subplots()
deltas = (registry['ecg_date']-registry['index_date']).dt.days
ax.hist(deltas[(deltas<30)&(deltas>-30)], bins=60, edgecolor='black', color='white')
ax.set_xlabel('ECG date - event date (days)')
ax.set_ylabel('# patients')
ax.set_xlim([-30, 30])
medn = np.median(deltas[(deltas<30)&(deltas>-30)])
# ax.set_ylim([0, 500])
# xticks = ax.get_xticklabels()
# xticks = [xticks[0]]+[matplotlib.text.Text(medn, 0, str(medn))]+xticks[1:]
# ax.set_xticks([0.0, medn, 20.0, 40.0, 60.0, 80.0, 100.0])
f.savefig('closest_ecg_date.png', dpi=500)


# %%
from sklearn.model_selection import train_test_split

train, valid = train_test_split(registry[['MRN', 'Toast']], test_size=0.3, train_size=0.7, stratify=registry['Toast'])
valid, test = train_test_split(valid, train_size=0.67, test_size=0.33, stratify=valid['Toast'])
train['MRN'].to_csv('/home/paolo/stroke_ecgs_one_per_patient/train.csv', index=False)
valid['MRN'].to_csv('/home/paolo/stroke_ecgs_one_per_patient/valid.csv', index=False)
test['MRN'].to_csv('/home/paolo/stroke_ecgs_one_per_patient/test.csv', index=False)


# %%
from ml4cvd.tensor_maps_partners_ecg_labels import TMAPS
tm = TMAPS['partners_ecg_afib_all_newest']
tm_read = TMAPS['partners_ecg_read_md']
registry['afib'] = 0.0
for i, patient in registry.iterrows():
    with h5py.File(f'/home/paolo/stroke_ecgs_one_per_patient/{int(patient["MRN"])}.hd5', 'r') as hd5:
        tt = tm.tensor_from_file(tm, hd5)
        registry.loc[registry['MRN'] == patient['MRN'], 'afib'] = np.argmax(tm.tensor_from_file(tm, hd5))
        read = tm_read.tensor_from_file(tm_read, hd5)
registry.to_csv('stroke_registry_mrn_mapped_toast_afib.csv', sep='\t', index=False)


# %%
from ml4cvd.tensor_maps_partners_ecg import TMAPS

hd5 = h5py.File('/home/paolo/stroke_ecgs_one_per_patient/4947104.hd5', 'r')
tm = TMAPS['toast_afib_newest']
tensor = tm.tensor_from_file(tm, hd5)
hd5.close()


# %%
hd5 = h5py.File('/home/paolo/stroke_ecgs_one_per_patient/881152.hd5', 'r')
tm = TMAPS['partners_ecg_2500_raw_newest']
tensor = tm.tensor_from_file(tm, hd5)
hd5.close()
tensor
# %%
registry.loc[registry['Toast']==6, 'Toast'] = 5
weights = {}
sum = 0.0
for i in range(1, 6):
    weights[str(i)] = len(registry)/len(registry[registry['Toast']==i])
    sum += weights[str(i)]
for i in range(1, 6):
    weights[str(i)]/=sum


# %%
# odds ratios
# scaled
import statsmodels.api as sm
explore = pd.read_csv('/home/paolo/stroke_ecg_predictions/explore/ecg_toast_predictions_features_age_sex_bmi_bp/tensors_all_intersect.csv')

baseline_keys = ['adult_gender_newest male',
                 'partners_ecg_age_newest',
                 'toast_bmi_newest',
                 'toast_sbp_newest',
                 'toast_dbp_newest'
                 ]
feature_keys = ['partners_feat_atrialrate_md',
       'partners_feat_paxis_md', 'partners_feat_poffset_md',
       'partners_feat_ponset_md', 'partners_feat_qoffset_md',
       'partners_feat_qonset_md', 'partners_feat_qrscount_md',
       'partners_feat_qrsduration_md', 'partners_feat_qtcorrected_md',
       'partners_feat_qtinterval_md', 'partners_feat_raxis_md',
       'partners_feat_taxis_md', 'partners_feat_toffset_md',
       'partners_feat_ventricularrate_md', 'toast_afib_newest']

feature_key_labels = ['Atrial rate', 'P axis', 'P offset', 'P onset', 
                     'Q offset', 'Q onset', '# QRS', 'QRS duration',
                     'QTc', 'QT', 'R axis', 'T axis', 'T offset', 'Ventricular rate',
                     'in Afib']

output_keys = ['toast_subtype_newest DefCE',
       'toast_subtype_newest PosCE', 'toast_subtype_newest LAA',
       'toast_subtype_newest SAO', 'toast_subtype_newest Undet']
or_dic = {}
from sklearn.preprocessing import StandardScaler
for output_key in output_keys:
    or_dic[output_key] = {}
    for feature_key in feature_keys:
        tmp_data = explore[baseline_keys + [feature_key, output_key]]
        tmp_data['intercept'] = 1.0
        tmp_data_val = tmp_data.values
        scaler = StandardScaler()
        tmp_data_norm = scaler.fit_transform(tmp_data_val)
        last_norm = -2
        if feature_key == 'toast_afib_newest':
            last_norm = -3
        tmp_data_val[:, 1:last_norm] = tmp_data_norm[:, 1:last_norm]
        res = sm.Logit(tmp_data[output_key], tmp_data[[feature_key, 'intercept']+baseline_keys]).fit()
        or_dic[output_key][feature_key] = {}
        or_dic[output_key][feature_key]['OR'] = np.exp(res.params.loc[feature_key])
        or_dic[output_key][feature_key]['CI'] = np.exp(res.conf_int().loc[feature_key].values)
        or_dic[output_key][feature_key]['p'] = res.pvalues[feature_key]
        or_dic[output_key][feature_key]['n'] = np.sum(tmp_data[output_key])


# %%
%matplotlib inline
f, ax = plt.subplots(1, 5)
f.set_size_inches(12, 4)
for i, output_key in enumerate(output_keys):
    ors = []
    cis_minus = []
    cis_plus = []
    labels = []
    for feature_key in feature_keys:
        if feature_key == 'toast_afib_newest':
            continue
        ors.append(or_dic[output_key][feature_key]['OR'])
        cis_minus.append(ors[-1]-or_dic[output_key][feature_key]['CI'][0])
        cis_plus.append(or_dic[output_key][feature_key]['CI'][1]-ors[-1])
#       
    ax[i].errorbar(ors, np.arange(len(ors)), xerr=(cis_minus, cis_plus), marker='o', linestyle='', color='black')
    ax[i].plot([1.0, 1.0], [-1.0, len(ors)], 'k--')
    ax[i].set_yticks(np.arange(len(feature_keys)))
    ax[i].set_yticklabels(feature_key_labels[:-1])
    if i > 0: ax[i].set_yticklabels([])
    ax[i].set_xscale('log', basex=np.exp(1))
    ax[i].set_xticks([0.0625, 0.25, 1, 4, 16])
    ax[i].set_xticklabels(map(str, ['', 0.25, 1, 4, 16]))
    
    ax[i].set_title(f'{output_key.split()[1]}\n n$_+$ = {int(or_dic[output_key][feature_key]["n"])} / {len(explore)}')
    ax[i].set_ylim([-1.0, len(ors)])
    ax[i].set_xlim([0.125, 20])
    plt.tight_layout()
f.text(0.5, 0.01, 'Odds ratio (per 1-SD increase)', ha='center')
f.savefig(f'ecg_features_or_test.png', dpi=500)


# %%
import seaborn as sns
roc_dic = {}
xlabels = ['age+sex+bmi+bp', 
           'ECG features',
           'ECG features+in Afib', 'ECG features+age+sex+bmi+bp', 'ECG waveform (CNN)']

roc_arr = np.array([[0.695, 0.526, 0.389, 0.452, 0.573], 
                   [0.726, 0.564, 0.605, 0.519, 0.707],
                   [0.811, 0.506, 0.565, 0.477, 0.650],                                     
                   [0.672, 0.489, 0.591, 0.464, 0.733],
                   [0.527, 0.578, 0.536, 0.459, 0.406]])

f, ax = plt.subplots()
f.set_size_inches(6, 3)
sns.heatmap(roc_arr, cmap='gray', annot=True, 
            xticklabels=[key.split()[1] for key in output_keys],
            yticklabels=xlabels)
plt.tight_layout()
f.savefig('roc_auc_toast.png', dpi=500)

# %%
