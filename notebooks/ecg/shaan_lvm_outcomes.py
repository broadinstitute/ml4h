# %%
import pandas as pd

individuals = pd.read_csv('/home/pdiachil/projects/shaan/outcomes_ukbb_sample_111020.csv', index_col=0)
ecg_reads = pd.read_csv('/home/pdiachil/projects/shaan/ECG_Views_LVH.csv', index_col=0)
ecg_reads['sample_id'] = ecg_reads['patient_id']
ecg_reads = ecg_reads.drop(columns=['patient_id'])
# %%
ecg_reads = ecg_reads.merge(individuals, on='sample_id')
ecg_reads[['sample_id', 'aVL', 'Sokolow_Lyon', 'Cornell', 'LVH', 'ecg_text']].to_csv('/home/pdiachil/projects/shaan/ecg_reads_lvh_sample_1111020.csv', index=False)
# %%
