# %%
import pandas as pd
import numpy as np

# load all tabular summary rows
tabular_summary = pd.concat([pd.read_csv(f'/home/paolo/exercise_ecgs/mgh-vm1/less_than_{d}m.csv') for d in range(3, 8)])
test_demographics = pd.read_csv('/home/paolo/exercise_ecgs/mgh-vm1/test_demographics.csv')
patient_demographics = pd.read_csv('/home/paolo/exercise_ecgs/mgh-vm1/patient_demographics.csv')

# %%
import re
import logging

def _clean_mrn(mrn: str) -> str:
    # TODO additional cleaning like o->0, |->1
    try:
        clean = re.sub(r'[^0-9]', '', mrn)
        clean = int(clean)
        if not clean:
            raise ValueError()
        return str(clean)
    except (ValueError, TypeError) as e:
        logging.info(f'Could not clean MRN "{mrn}" to an int. Using "bad_mrn".')
        return 'bad_mrn'
# %%
tabular_tests = pd.DataFrame({'TestID': pd.unique(tabular_summary['TestID'])})
tabular_tests = tabular_tests.merge(test_demographics, on='TestID')
tabular_tests = tabular_tests.merge(patient_demographics, on='TestID')
tabular_tests['MRN'] = tabular_tests['PatientID'].apply(_clean_mrn)
tabular_tests['Date_Time'] = pd.to_datetime(tabular_tests['AcquisitionDateTime_DT'])
tabular_tests['Date_Time'] = tabular_tests['Date_Time'].dt.round('D')
tabular_tests = tabular_tests[['MRN', 'Date_Time']].drop_duplicates()

# %%
def fpath_to_mrn(fpath: str) -> str:
    mrn = fpath.split('/')[-1].replace('.hd5', '')
    return mrn
rest_ecgs = pd.read_csv('/home/paolo/exercise_ecgs/tensors_all_union.csv', usecols=['fpath', 'partners_ecg_datetime'])
rest_ecgs['Date_Time'] = pd.to_datetime(rest_ecgs['partners_ecg_datetime'])
rest_ecgs['Date_Time'] = rest_ecgs['Date_Time'].dt.round('D')
rest_ecgs['MRN'] = rest_ecgs['fpath'].apply(fpath_to_mrn)
rest_ecgs = rest_ecgs[['MRN', 'Date_Time']].drop_duplicates()

# %%
rest_exercise_ecgs = rest_ecgs.merge(tabular_tests, on='MRN', suffixes=('_rest', '_exercise'))
rest_exercise_ecgs['Date_Time_diff'] = rest_exercise_ecgs['Date_Time_exercise'] - rest_exercise_ecgs['Date_Time_rest']
rest_exercise_ecgs[(rest_exercise_ecgs['Date_Time_diff']>=pd.Timedelta('0 days')) &\
                   (rest_exercise_ecgs['Date_Time_diff']<=pd.Timedelta('7 days'))][['MRN', 'Date_Time_exercise']].drop_duplicates()
# rest_exercise_ecgs[(rest_exercise_ecgs['Date_Time_diff']>=pd.Timedelta('0 days')) &\
#                    (rest_exercise_ecgs['Date_Time_diff']<=pd.Timedelta('30 days'))][['MRN']].drop_duplicates()

# %%
import matplotlib.pyplot as plt
import seaborn as sns
df_30d = rest_exercise_ecgs[(rest_exercise_ecgs['Date_Time_diff']>=pd.Timedelta('0 days')) &\
                            (rest_exercise_ecgs['Date_Time_diff']<=pd.Timedelta('30 days'))].groupby(by=['MRN', 'Date_Time_exercise']).min()
f, ax = plt.subplots()
f.set_size_inches(4, 3)
sns.distplot(df_30d['Date_Time_diff'].dt.days, kde=False, bins=range(31), ax=ax, color='black')
ax.set_xlim([0, 30])
ax.set_xlabel('date exercise - date rest (days)')
ax.set_ylabel('counts')
plt.tight_layout()
f.savefig('rest_diff_exercise.png', dpi=500)

# %%
df_7d = rest_exercise_ecgs[(rest_exercise_ecgs['Date_Time_diff']>=pd.Timedelta('0 days')) &\
                           (rest_exercise_ecgs['Date_Time_diff']<=pd.Timedelta('7 days'))].groupby(by=['MRN', 'Date_Time_exercise']).min()
df_7d = df_7d.reset_index()

df_30d = rest_exercise_ecgs[(rest_exercise_ecgs['Date_Time_diff']>=pd.Timedelta('0 days')) &\
                           (rest_exercise_ecgs['Date_Time_diff']<=pd.Timedelta('30 days'))].groupby(by=['MRN', 'Date_Time_exercise']).min()
df_30d = df_30d.reset_index()

tabular_tests = pd.DataFrame({'TestID': pd.unique(tabular_summary['TestID'])})
tabular_tests = tabular_tests.merge(test_demographics, on='TestID')
tabular_tests = tabular_tests.merge(patient_demographics, on='TestID')
tabular_tests['MRN'] = tabular_tests['PatientID'].apply(_clean_mrn)
tabular_tests['Date_Time'] = pd.to_datetime(tabular_tests['AcquisitionDateTime_DT'])
tabular_tests['Date_Time'] = tabular_tests['Date_Time'].dt.round('D')
tabular_tests = tabular_tests[['TestID', 'MRN', 'Date_Time']]
tabular_tests = tabular_tests.groupby(['MRN', 'Date_Time']).min().reset_index()

df_7d = df_7d[['MRN', 'Date_Time_exercise', 'Date_Time_diff']]
df_7d = df_7d.merge(tabular_tests, left_on=['MRN', 'Date_Time_exercise'], right_on=['MRN', 'Date_Time'])
df_7d_summary = tabular_summary.merge(df_7d, on='TestID')
df_7d_summary['Date_Time_diff'] = df_7d_summary['Date_Time_diff'].dt.days

df_30d = df_30d[['MRN', 'Date_Time_exercise', 'Date_Time_diff']]
df_30d = df_30d.merge(tabular_tests, left_on=['MRN', 'Date_Time_exercise'], right_on=['MRN', 'Date_Time'])
df_30d_summary = tabular_summary.merge(df_30d, on='TestID')
df_30d_summary['Date_Time_diff'] = df_30d_summary['Date_Time_diff'].dt.days

# %%
# Filter by METS
df_7d_mets_max = df_7d_summary.groupby(by='TestID').max()
df_7d_mets_min = df_7d_summary.groupby(by='TestID').min()
df_7d_mets_mean = df_7d_summary.groupby(by='TestID').mean()
did_exercise_7d = df_7d_mets_max['METS']>4.0001

from scipy.stats import pearsonr
def pearsonr_pval(x,y):
    return pearsonr(x,y)[1]<0.05

def asterisk_if_false(x):
    return '*' if x < 0.05 else ''

labels = ['# measurements',
          'Test duration',
          'Max phaseID',
          'Max stageID',
          'Max speed',
          'Max grade',
          'Max METS',
          'Max HR',
          'Max SBP',
          'Max DBP',
          'Delay from 12L'
]

f, ax = plt.subplots()
f.set_size_inches(5, 4)
sns.heatmap(df_7d_mets_max[did_exercise_7d].corr(), ax=ax, cmap='RdGy', center=0.0, fmt='.2f', cbar=True)
ax.set_xticklabels(labels, rotation=45, ha='right')
ax.set_yticklabels(labels)
plt.tight_layout()
f.savefig('heatmap_tabular_summary.png', dpi=500)

f, ax = plt.subplots()
f.set_size_inches(5, 4)
sns.heatmap(df_7d_mets_max[did_exercise_7d].corr(method=pearsonr_pval), ax=ax, annot=True, fmt='.2f', cbar=True)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
plt.tight_layout()
f.savefig('heatmap_tabular_summary_pvals.png', dpi=500)

# %%
# Filter by METS
df_30d_mets_max = df_30d_summary.groupby(by='TestID').max()
df_30d_mets_min = df_30d_summary.groupby(by='TestID').min()
df_30d_mets_mean = df_30d_summary.groupby(by='TestID').mean()
did_exercise_30d = df_30d_mets_max['METS']>4.0001

from scipy.stats import pearsonr
def pearsonr_pval(x,y):
    return pearsonr(x,y)[1]<0.05

def asterisk_if_false(x):
    return '*' if x < 0.05 else ''

labels = ['# measurements',
          'Test duration',
          'Max phaseID',
          'Max stageID',
          'Max speed',
          'Max grade',
          'Max METS',
          'Max HR',
          'Max SBP',
          'Max DBP',
          'Delay from 12L'
]

f, ax = plt.subplots()
f.set_size_inches(5, 4)
sns.heatmap(df_30d_mets_max[did_exercise_30d].corr(), ax=ax, cmap='RdGy', center=0.0, fmt='.2f', cbar=True)
ax.set_xticklabels(labels, rotation=45, ha='right')
ax.set_yticklabels(labels)
plt.tight_layout()
f.savefig('heatmap_tabular_summary_30d.png', dpi=500)

f, ax = plt.subplots()
f.set_size_inches(5, 4)
sns.heatmap(df_30d_mets_max[did_exercise_30d].corr(method=pearsonr_pval), ax=ax, annot=True, fmt='.2f', cbar=True)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
plt.tight_layout()
f.savefig('heatmap_tabular_summary_pvals_30d.png', dpi=500)

# %%
def _clean_mgb_mrn(mrn: str) -> str:
    # TODO additional cleaning like o->0, |->1
    try:
        clean = int(mrn)
        if not clean:
            raise ValueError()
        return str(clean)
    except (ValueError, TypeError) as e:
        logging.info(f'Could not clean MRN "{mrn}" to an int. Using "bad_mrn".')
        return 'bad_mrn'

mgb_patients = pd.read_csv('/home/paolo/exercise_ecgs/all_id.txt', sep='\t')
mgb_patients['MGH_MRN'] = mgb_patients['MGH_MRN'].apply(_clean_mgb_mrn)
mgb_patients = mgb_patients[mgb_patients['MGH_MRN']!='bad_mrn']
mgb_patients['MRN'] = mgb_patients['MGH_MRN']

rest_exercise_ecgs_genetics = rest_exercise_ecgs.merge(mgb_patients['MRN'], on='MRN')

rest_exercise_ecgs_genetics[(rest_exercise_ecgs_genetics['Date_Time_diff']>=pd.Timedelta('0 days')) &\
                            (rest_exercise_ecgs_genetics['Date_Time_diff']<pd.Timedelta('30 days'))][['MRN', 'Date_Time_rest']].drop_duplicates()
# rest_exercise_ecgs_genetics[(rest_exercise_ecgs_genetics['Date_Time_diff']>=pd.Timedelta('0 days')) &\
#                             (rest_exercise_ecgs_genetics['Date_Time_diff']<=pd.Timedelta('30 days'))][['MRN']].drop_duplicates()
# %%
print(len(df_30d.merge(did_exercise_30d[did_exercise_30d].reset_index()['TestID'], on='TestID')))
print(len(np.unique(df_30d.merge(did_exercise_30d[did_exercise_30d].reset_index()['TestID'], on='TestID')['MRN'])))

# %%
def did_exercise_for_1m(df):
    exercise = df[df['METS']>=4.0]
    exercise_time = exercise['TimeFromTestStart'].max() - exercise['TimeFromTestStart'].min()
    return exercise_time > 60000
df_7d_summary_group = df_7d_summary.groupby(by='TestID')
df_30d_summary_group = df_30d_summary.groupby(by='TestID')
exercise_for_1m_7d = df_7d_summary_group.apply(did_exercise_for_1m)
exercise_for_1m_30d = df_30d_summary_group.apply(did_exercise_for_1m)

# %%
def last_mets_equals_1(df):
    arg_max_n = np.argmax(df['NthOccur'])
    guess_rec_mets = df.iloc[arg_max_n]['METS']
    return np.abs(guess_rec_mets-1.0) < 1e-3
last_mets_1_7d = df_7d_summary_group.apply(last_mets_equals_1)
last_mets_1_30d = df_30d_summary_group.apply(last_mets_equals_1)

# %%
print(np.sum(np.logical_and(last_mets_1_7d, did_exercise_7d)))
print(len(np.unique(df_7d.merge(last_mets_1_7d[np.logical_and(last_mets_1_7d, did_exercise_7d)].reset_index()['TestID'], on='TestID')['MRN'])))

print(np.sum(np.logical_and(last_mets_1_30d, did_exercise_30d)))
print(len(np.unique(df_30d.merge(last_mets_1_30d[np.logical_and(last_mets_1_30d, did_exercise_30d)].reset_index()['TestID'], on='TestID')['MRN'])))

# %%
print(np.sum(exercise_for_1m_7d))
print(len(np.unique(df_7d.merge(exercise_for_1m_7d[exercise_for_1m_7d].reset_index()['TestID'], on='TestID')['MRN'])))

print(np.sum(exercise_for_1m_30d))
print(len(np.unique(df_30d.merge(exercise_for_1m_30d[exercise_for_1m_30d].reset_index()['TestID'], on='TestID')['MRN'])))

print(np.sum(np.logical_and(last_mets_1_7d, exercise_for_1m_7d)))
print(len(np.unique(df_7d.merge(last_mets_1_7d[np.logical_and(last_mets_1_7d, exercise_for_1m_7d)].reset_index()['TestID'], on='TestID')['MRN'])))

print(np.sum(np.logical_and(last_mets_1_30d, exercise_for_1m_30d)))
print(len(np.unique(df_30d.merge(last_mets_1_30d[np.logical_and(last_mets_1_30d, exercise_for_1m_30d)].reset_index()['TestID'], on='TestID')['MRN'])))

# %%
def test_duration(df):
    arg_0 = np.argmin(df['NthOccur'])
    arg_n = np.argmax(df['NthOccur'])
    return df.iloc[arg_n]['TimeFromTestStart'] - df.iloc[arg_0]['TimeFromTestStart']

test_duration_7d = df_7d_summary.groupby('TestID').apply(test_duration)
#validated_test_duration_7d = validated_tests_7d.groupby('TestID').apply(test_duration)

# %%
f, ax = plt.subplots()
sns.distplot(test_duration_7d/1000.0, ax=ax, kde=False, bins=range(0, 6000, 100), color='gray', label='Before QC')
sns.distplot(validated_test_duration_7d/1000.0, ax=ax, kde=False, bins=range(0, 6000, 100), color='black', label='After QC')
ax.set_xlim([0.0, 6000.0])
ax.legend()
ax.set_xlabel('Test duration (s)')
ax.set_ylabel('Counts')
plt.tight_layout()
f.savefig('test_duration_qc.png', dpi=500)

# %%
for feat, lims in zip(['METS', 'Speed', 'Grade'],
                      [(1, 15, 0.5), (0, 5, 0.1), (0, 15, 0.5)]):        
    average_feat_7d = df_7d_summary.groupby('TestID').mean()[feat]
    validated_average_feat_7d = validated_tests_7d.groupby('TestID').mean()[feat]

    f, ax = plt.subplots()
    sns.distplot(validated_average_feat_7d, ax=ax, kde=False, bins=np.arange(*lims), color='black', label='After QC')
    sns.distplot(average_feat_7d, ax=ax, kde=False, bins=np.arange(*lims), color='gray', label='Before QC')
    ax.set_xlim([lims[0], lims[1]])
    ax.legend()
    ax.set_xlabel(f'Average {feat}')
    ax.set_ylabel('Counts')
    plt.tight_layout()
    f.savefig(f'{feat}_qc.png', dpi=500)

# %%
def hrr_simple(df, deltat=50.0):
    arg_hr = np.argmax(df['HeartRate'])
    max_hr = df.iloc[arg_hr]['HeartRate']
    max_t = df.iloc[arg_hr]['TimeFromTestStart']
    arg_hrr = np.argmin(np.abs(df['TimeFromTestStart']-(max_t+deltat*1000.0)))
    return max_hr - df.iloc[arg_hrr]['HeartRate']

def hrr_simple_phase(df, deltat=50.0):
    arg_hr = np.argmax(df['HeartRate'])
    max_hr = df.iloc[arg_hr]['HeartRate']
    arg_n = np.argmax(df['NthOccur'])
    guess_rec_phase = df.iloc[arg_n]['PhaseID']
    t_rec = np.min(df[df['PhaseID']==guess_rec_phase]['TimeFromTestStart'])
    arg_hrr = np.argmin(np.abs(df['TimeFromTestStart']-(t_rec+deltat*1000.0)))
    return max_hr - df.iloc[arg_hrr]['HeartRate']

validated_tests_7d = df_7d_summary.merge(last_mets_1_7d[np.logical_and(last_mets_1_7d, exercise_for_1m_7d)].reset_index()['TestID'], on='TestID')
validated_tests_30d = df_30d_summary.merge(last_mets_1_30d[np.logical_and(last_mets_1_30d, exercise_for_1m_30d)].reset_index()['TestID'], on='TestID')

hrr_simple_7d = validated_tests_7d.groupby('TestID').apply(hrr_simple)
hrr_simple_30d = validated_tests_30d.groupby('TestID').apply(hrr_simple)

hrr_phase_7d = validated_tests_7d.groupby('TestID').apply(hrr_simple_phase)
hrr_phase_30d = validated_tests_30d.groupby('TestID').apply(hrr_simple_phase)

# %%
def hrr_phase_within_1min(df, deltat=50.0):
    arg_n = np.argmax(df['NthOccur'])
    guess_rec_phase = df.iloc[arg_n]['PhaseID']
    t_rec = np.min(df[df['PhaseID']==guess_rec_phase]['TimeFromTestStart'])
    max_hr = np.max(df[((t_rec-df['TimeFromTestStart'])<60000.0) & ((t_rec-df['TimeFromTestStart'])>0.0)]['HeartRate'])
    arg_hrr = np.argmin(np.abs(df['TimeFromTestStart']-(t_rec+deltat*1000.0)))
    return max_hr - df.iloc[arg_hrr]['HeartRate']

hrr_phase_1min_7d = validated_tests_7d.groupby('TestID').apply(hrr_phase_within_1min)
hrr_phase_1min_30d = validated_tests_30d.groupby('TestID').apply(hrr_phase_within_1min)

# %%
def hrr_phase_median_within_1min(df, deltat=50.0):
    arg_n = np.argmax(df['NthOccur'])
    guess_rec_phase = df.iloc[arg_n]['PhaseID']
    t_rec = np.min(df[df['PhaseID']==guess_rec_phase]['TimeFromTestStart'])
    try:
        argmax_hr = np.argmax(df[((t_rec-df['TimeFromTestStart'])<60000.0) & ((t_rec-df['TimeFromTestStart'])>0.0)]['HeartRate'])
    except:
        return -1
    t_hrmax = df[((t_rec-df['TimeFromTestStart'])<60000.0) & ((t_rec-df['TimeFromTestStart'])>0.0)].iloc[argmax_hr]['TimeFromTestStart']
    max_hr = np.median(df[np.abs((t_hrmax-df['TimeFromTestStart']))<3000.0]['HeartRate'])
    arg_hrr = np.argmin(np.abs(df['TimeFromTestStart']-(t_rec+deltat*1000.0)))
    return max_hr - df.iloc[arg_hrr]['HeartRate']

hrr_phase_median_1min_7d = validated_tests_7d.groupby('TestID').apply(hrr_phase_median_within_1min)
hrr_phase_median_1min_30d = validated_tests_30d.groupby('TestID').apply(hrr_phase_median_within_1min)

# %%
def hrr_phase_within_30s(df, deltat=50.0):
    arg_n = np.argmax(df['NthOccur'])
    guess_rec_phase = df.iloc[arg_n]['PhaseID']
    t_rec = np.min(df[df['PhaseID']==guess_rec_phase]['TimeFromTestStart'])
    max_hr = np.max(df[((t_rec-df['TimeFromTestStart'])<30000.0) & ((t_rec-df['TimeFromTestStart'])>0.0)]['HeartRate'])
    arg_hrr = np.argmin(np.abs(df['TimeFromTestStart']-(t_rec+deltat*1000.0)))
    return max_hr - df.iloc[arg_hrr]['HeartRate']

hrr_phase_30s_7d = validated_tests_7d.groupby('TestID').apply(hrr_phase_within_30s)
hrr_phase_30s_30d = validated_tests_30d.groupby('TestID').apply(hrr_phase_within_30s)


# %%
f, ax = plt.subplots()
#sns.distplot(hrr_simple_7d, ax=ax, kde=False, bins=np.arange(0, 100, 1), color=[0.8, 0.8, 0.8], label='1st definition')
#sns.distplot(hrr_phase_7d, ax=ax, kde=False, bins=np.arange(0, 100, 1), color=[0.6, 0.6, 0.6], label='2nd definition')
sns.distplot(hrr_phase_1min_7d, ax=ax, kde=False, bins=np.arange(0, 101, 1), color=[0.4, 0.4, 0.4], label='3rd definition')
#sns.distplot(hrr_phase_median_1min_7d, ax=ax, kde=False, bins=np.arange(0, 100, 1), color=[0.2, 0.2, 0.2], label='4th definition')
sns.distplot(hrr_phase_30s_7d, ax=ax, kde=False, bins=np.arange(0, 101, 1), color=[0.2, 0.2, 0.2], label='4th definition')
ax.legend()
ax.set_xlabel('HRR (bpm)')
ax.set_ylabel('Counts')
plt.tight_layout()
f.savefig('hrr_defs_7d.png', dpi=500)

# %%
f, ax = plt.subplots()
sns.distplot(hrr_simple_30d, ax=ax, kde=False, bins=np.arange(0, 100, 1), color=[0.8, 0.8, 0.8], label='1st definition')
sns.distplot(hrr_phase_30d, ax=ax, kde=False, bins=np.arange(0, 100, 1), color=[0.5, 0.5, 0.5], label='2nd definition')
sns.distplot(hrr_phase_1min_30d, ax=ax, kde=False, bins=np.arange(0, 100, 1), color=[0.2, 0.2, 0.2], label='3rd definition')
ax.legend()
ax.set_xlabel('HRR (bpm)')
ax.set_ylabel('Counts')
plt.tight_layout()
f.savefig('hrr_defs_30d.png', dpi=500)

# %%
validated_tests_7d.to_csv('validated_tests_7d.csv', index=False)
# %%
def max_hr_to_rec(df):
    arg_n = np.argmax(df['NthOccur'])
    guess_rec_phase = df.iloc[arg_n]['PhaseID']
    t_rec = np.min(df[df['PhaseID']==guess_rec_phase]['TimeFromTestStart'])
    pre_rec = ((t_rec-df['TimeFromTestStart'])<60000.0) & ((t_rec-df['TimeFromTestStart'])>0.0)
    try:
        n_max_hr = np.argmax(df[pre_rec]['HeartRate'])
    except ValueError:
        print(df.iloc[0]['TestID'])
        return -1
    t_max_hr = df[pre_rec].iloc[n_max_hr]['TimeFromTestStart']
    return t_rec - t_max_hr

t_max_hr_7d = validated_tests_7d.groupby('TestID').apply(max_hr_to_rec)

# %%
f, ax = plt.subplots()
sns.distplot(t_max_hr_7d/1000.0, kde=False, color='gray', label='1st definition')
ax.set_ylabel('Counts')
ax.set_xlabel('Time from max HR and start of recovery (s)')
f.savefig('tmaxhr.png', dpi=500)
# %%
import numpy as np
def plot_exercise_ecg(df, xlim=None):
    f, ax = plt.subplots()
    f.set_size_inches(4, 3)
    ax2 = ax.twinx()
    ax.plot(df['TimeFromTestStart']/1000.0, df['METS'], 'd', markersize=10, color='gray')
    ax2.plot(df['TimeFromTestStart']/1000.0, df['HeartRate'], 'x', markersize=10, color='black')
    ax.set_xlabel('Time from start (s)')
    ax2.set_ylabel('Heart rate (bpm)')
    ax.set_ylabel('METS')
    if xlim:
        ax2.set_xlim(xlim)        
    plt.tight_layout()
    f.savefig(f'ex_hr_mets_{df["TestID"].values[0]}.png', dpi=500)

    f, ax = plt.subplots()
    f.set_size_inches(4, 3)
    ax2 = ax.twinx()
    ax.plot(df['TimeFromTestStart']/1000.0, df['PhaseID'], 'd', markersize=10, color='gray')
    ax2.plot(df['TimeFromTestStart']/1000.0, df['HeartRate'], 'x', markersize=10, color='black')
    ax.set_xlabel('Time from start (s)')
    ax2.set_ylabel('Heart rate (bpm)')
    ax.set_ylabel('PhaseID')
    plt.tight_layout()
    f.savefig(f'hrr0_hr_phase_{df["TestID"].values[0]}.png', dpi=500)

test_ids = [3000397]

for i, row in enumerate(test_ids):
    plot_exercise_ecg(validated_tests_7d[validated_tests_7d['TestID']==row])
# %%
tabular_30d = df_30d.merge(tabular_summary, on='TestID')
mins = tabular_30d.groupby(['TestID']).min()
maxs = tabular_30d.groupby(['TestID']).max()
diffs = pd.DataFrame()
diffs['TestDuration'] = maxs['TimeFromTestStart'] - mins['TimeFromTestStart']
# %%
f, ax = plt.subplots()
sns.distplot(diffs[diffs['TestDuration']< 1000000.0]]['TestDuration']/1000.0, kde=False, ax=ax, bins=1)
ax.set_xlim([0, 1000])

# %%
patients = pd.read_csv('/home/paolo/exercise_ecgs/mgh-vm1/ex_ecg_patientids_testids.csv')
patients['patientid'] = pd.to_numeric(patients['PatientID'], errors='coerce')
patients = patients.dropna()
patients['patientid'] = patients['patientid'].apply(int)
c3po_patients_fu = pd.read_csv('/home/paolo/mgh_mrns_to_extract/mgh_patients.csv', sep='\t')
c3po_patients = pd.read_csv('/home/paolo/mgh_mrns_to_extract/lc_outcomes.csv', sep=',')

# %%
patients.merge(c3po_patients_fu, on='patientid').iloc[-20:]
# %%
patients.merge(c3po_patients, left_on='patientid', right_on='Mrn')
# %%
import matplotlib.pyplot as plt


test = pd.read_csv('/home/paolo/exercise_ecgs/mgh-vm1/4213242.csv')
plot_exercise_ecg(test)

test = pd.read_csv('/home/paolo/exercise_ecgs/mgh-vm1/5976918.csv')
plot_exercise_ecg(test)

test = pd.read_csv('/home/paolo/exercise_ecgs/mgh-vm1/5176540.csv')
plot_exercise_ecg(test)

test = pd.read_csv('/home/paolo/exercise_ecgs/mgh-vm1/5848498.csv')
plot_exercise_ecg(test)
# %%
