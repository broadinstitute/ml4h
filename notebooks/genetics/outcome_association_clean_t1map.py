# %%
import numpy as np
import pandas as pd
from outcome_association_utils import odds_ratios, hazard_ratios, plot_or_hr, unpack_disease, regression_model, random_forest_model, plot_rsquared_covariates

# %%
########### Pretest exercise ECGs ###########################
# Read phenotype and covariates
phenotypes = pd.read_csv('/home/pdiachil/projects/t1map/pheno.txt', sep='\t')
phenotypes['sample_id'] = phenotypes['IID']
phenotypes['instance'] = 2
covariates = pd.read_csv('/home/pdiachil/projects/t1map/t1map_covariate_disease_serial_mri.csv')
phenotypes_covariates = phenotypes.merge(covariates, on=['sample_id', 'instance'], suffixes=('', '_dup'))
phenotypes_covariates['ivs_lvbp'] = phenotypes_covariates['ivs'] / phenotypes_covariates['lvbp']

# %%

phenos_to_binarize = []
phenos_to_binarize = []

for pheno in phenos_to_binarize:
    phenotypes[f'{pheno}_binary'] = (phenotypes[pheno] > phenotypes[pheno].quantile(0.80)).apply(float)

label_dic = {
    'age_mri': ['age', 'yrs'],
    'sex': ['male', ''],
    'height_mri': ['height', 'cm'],
    'weight_mri': ['weight', 'kg'],
    'cholesterol': ['cholesterol', 'mmol/L'],
    'ldl': ['ldl', 'mmol/L'],
    'fw': ['T1 FW', 'ms'],
    'ivs': ['T1 IVS', 'ms'],
    'lvbp': ['T1 LV BP', 'ms'],
    'rvbp': ['T1 RV BP', 'ms'],
    'ivs_inr': ['T1 IVS inr', ''],
    'ivs_lvbp': ['T1 IVS normalized', ''],
    'ivs_no_outliers': ['T1 IVS no outliers', '']
}

dont_scale = [
    'sex'
]
# %%
# Read diseases and unpack

disease_list = [
    ['atrial_fibrillation', 'AF'],
    ['hcm', 'HCM'],
    ['dcm', 'DCM'],
    ['hf', 'HF'],
    ['diabetes', 'DM'],
    ['ventricular_arrhythmia', 'ventricular arr.'],    
    ['hypertension', 'hypertension'],
    ['ckd', 'CKD'],
    ['aortic_stenosis', 'AS']
]

for disease, disease_label in disease_list:
    phenotypes_covariates.loc[phenotypes_covariates[f'{disease}_incident']>0.5, f'{disease}_censor_date'] = \
        phenotypes_covariates.loc[phenotypes_covariates[f'{disease}_incident']>0.5, f'date_incident_{disease}']
    
    phenotypes_covariates[f'{disease}_censor_date'] = pd.to_datetime(phenotypes_covariates[f'{disease}_censor_date'])

    phenotypes_covariates.loc[phenotypes_covariates[f'{disease}_prevalent']>0.5, f'{disease}_censor_date'] = \
        pd.to_datetime(phenotypes_covariates.loc[phenotypes_covariates[f'{disease}_prevalent']>0.5, f'date_mri']) - pd.to_timedelta('1 day')

phenotypes_covariates['instance2_date'] = pd.to_datetime(phenotypes_covariates['date_mri'])
# %%
covariates = ['age_mri', 'sex', 'height_mri', 'weight_mri']
phenos = ['ivs_inr']
labels = {key: label_dic[key] for key in phenos}
phenotypes_covariates = phenotypes_covariates.dropna(subset=covariates+phenos)
odds_ratio_multivariable = odds_ratios(
    phenotypes_covariates, phenotypes_covariates, labels,
    disease_list, covariates=covariates, instance=2, dont_scale=dont_scale,
)
plot_or_hr(odds_ratio_multivariable, label_dic, disease_list, f'or_multivariate_t1map', occ='prevalent', horizontal_line_y=0)

hazard_ratio_multivariable = hazard_ratios(
    phenotypes_covariates, phenotypes_covariates, labels,
    disease_list, covariates=covariates, instance=2, dont_scale=dont_scale,
)
plot_or_hr(hazard_ratio_multivariable, label_dic, disease_list, f'hr_multivariate_t1map', occ='incident', horizontal_line_y=0)

# %%
# Disease centric figure
import matplotlib.pyplot as plt

def plot_or_hr_disease(or_dic, label_dic, disease_list, occ='prevalent', horizontal_line_y=None):
    ratio_type = 'OR' if 'prevalent' in occ else 'HR'
    ratio_label = 'Odds' if 'prevalent' in occ else 'Hazard'
    ors = []
    cis_minus = []
    cis_plus = []
    labels = []
    ors_cis_plus = []
    ors_cis_minus = []
    for dis, dis_label in disease_list[::-1]:        
        
        ors.append(np.exp(np.log(or_dic[f'{dis}_{occ}'][ratio_type])))
        cis_minus.append(np.exp(np.log(or_dic[f'{dis}_{occ}'][ratio_type]-or_dic[f'{dis}_{occ}']['CI'][0])))
        cis_plus.append(np.exp(np.log(or_dic[f'{dis}_{occ}']['CI'][1]-or_dic[f'{dis}_{occ}'][ratio_type])))
        ors_cis_plus.append(or_dic[f'{dis}_{occ}']['CI'][1])
        ors_cis_minus.append(or_dic[f'{dis}_{occ}']['CI'][0])
        label = dis_label + '  '
        if or_dic[dis+"_"+occ]['p'] < 0.05:
            label += '*'
        # label += f'\t(n$_+$ = {int(or_dic[dis+"_"+occ]["n"]):>6} / {int(or_dic[dis+"_"+occ]["ntot"])})'
        labels.append(label)

    f, ax = plt.subplots()
    f.set_size_inches(4, 4)
    ax.errorbar(ors, np.arange(len(ors)), xerr=(cis_minus, cis_plus), marker='o', linestyle='', color='black')
    ax.plot([1.0, 1.0], [-1.0, len(ors)], 'k-')
    if horizontal_line_y:
        ax.plot([-20.0, 20.0], [horizontal_line_y, horizontal_line_y], 'k:')
    ax.set_yticks(np.arange(len(ors)))
    ax.set_yticklabels(labels)
    ax.set_xscale('log', basex=np.exp(1))
    ax.set_xticks([0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 8.0])
    ax.set_xticklabels(map(str, [0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 8.0]))
    ax.set_xlabel(f'{ratio_label} ratio\n(per 1-SD increase in IVS T1 rank)')
    ax.set_ylim([-1.0, len(ors)])
    ax.set_xlim([min(1.0, min(ors_cis_minus)*0.99), max(2.0, max(ors_cis_plus)*1.01)])
    f.tight_layout()
    f.savefig(f't1map_{occ}.png', dpi=500)

plot_or_hr_disease(odds_ratio_multivariable['ivs_inr'], label_dic, disease_list)
plot_or_hr_disease(hazard_ratio_multivariable['ivs_inr'], label_dic, disease_list, occ='incident')
