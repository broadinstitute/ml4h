# %%
import numpy as np
import pandas as pd
from outcome_association_utils import odds_ratios, hazard_ratios, plot_or_hr, unpack_disease, regression_model, random_forest_model, plot_rsquared_covariates

# %%
########### Pretest exercise ECGs ###########################
# Read phenotype and covariates
phenotypes = pd.read_csv('/home/pdiachil/ml/notebooks/genetics/la_volumes_petersen_covariates.csv')
phenotypes = phenotypes[phenotypes['LA_poisson_cleaned_min']<150]

phenos_to_binarize = ['LA_poisson_cleaned_max', 'LA_poisson_cleaned_min', 'LA_Biplan_vol_max', 'LA_Biplan_vol_min']
# phenos_to_binarize = ['LA_poisson_cleaned_max', 'LA_poisson_cleaned_min']
for pheno in phenos_to_binarize:
    phenotypes[f'{pheno}_binary'] = (phenotypes[pheno] > phenotypes[pheno].quantile(0.80)).apply(float)

label_dic = {
    'LA_Biplan_vol_max': ['LA$_{max}$ (biplane)', 'ml'],
    'LA_Biplan_vol_min': ['LA$_{min}$ (biplane)', 'ml'],
    'LA_poisson_cleaned_max': ['LA$_{max}$ (3-D surf)', 'ml'],
    'LA_poisson_cleaned_min': ['LA$_{min}$ (3-D surf)', 'ml'],
    'LA_Biplan_vol_max_binary': ['enlarged LA$_{max}$ (biplane)', ''],
    'LA_Biplan_vol_min_binary': ['enlarged LA$_{min}$ (biplane)', ''],      
    'LA_poisson_cleaned_max_binary': ['enlarged LA$_{max}$ (3-D surf)', ''],     
    'LA_poisson_cleaned_min_binary': ['enlarged LA$_{min}$ (3-D surf)', ''],    
    'resting_hr': ['Rest HR', 'beats'],
    'age': ['Age', 'yrs'],
    'male': ['Male', ''],
    # 'nonwhite': ['Nonwhite', ''],
    'bmi': ['BMI', 'units'],
    'cholesterol': ['Cholesterol', 'mmol/L'],
    'HDL': ['HDL', 'mmol/L'],
    'current_smoker': ['Current smoker', ''],
    # 'diastolic_bp': ['Diastolic blood pressure', 'mmHg'],
    'systolic_bp': ['Systolic blood pressure', 'mmHg'],
    # 'gfr': ['eGFR', 'mL/min/1.73 m2'],
    # 'creatinine': ['Creatinine', 'umol/L'],
    'c_lipidlowering': ['Lipid lowering drugs', ''],
    'c_antihypertensive': ['Antihypertensive drugs', '']
}

dont_scale = ['male', 'nonwhite', 'current_smoker', 'c_lipidlowering', 'c_antihypertensive', 'LA_poisson_cleaned_max_binary', 'LA_poisson_cleaned_min_binary', 'LA_Biplan_vol_min_binary', 'LA_Biplan_vol_max_binary']
phenotypes = phenotypes.dropna(subset=['age', 'male'])

# %%
import matplotlib.pyplot as plt
import seaborn as sns
f, ax = plt.subplots()
sns.distplot(phenotypes['LA_poisson_cleaned_min'])
phenotypes['LA_poisson_cleaned_min'].min()
# %%
# Read diseases and unpack
diseases = pd.read_csv('/home/pdiachil/ml/notebooks/genetics/bq_diseases.tsv', sep='\t')
diseases['censor_date'] = pd.to_datetime(diseases['censor_date'])

disease_list = [['Atrial_fibrillation_or_flutter_v2', 'atrial fibrillation'],
                ]
diseases_unpack = unpack_disease(diseases, disease_list, phenotypes)



# %%
# Univariate
odds_ratio_univariable = odds_ratios(phenotypes, diseases_unpack, label_dic,
                                     disease_list, covariates=[], instance=0, dont_scale=dont_scale)
plot_or_hr(odds_ratio_univariable, label_dic, disease_list, f'or_univariate_pretest')

hazard_ratio_univariable = hazard_ratios(phenotypes, diseases_unpack, label_dic,
                                         disease_list, covariates=[], instance=0, dont_scale=dont_scale)

plot_or_hr(hazard_ratio_univariable, label_dic, disease_list, f'hr_univariate_pretest', occ='incident')

# %%
# Univariate lighter
phenotype_subset = ['50_hrr_actual', '50_hrr_downsample_augment_prediction', 'resting_hr',
                    '50_hrr_actual_binary', '50_hrr_downsample_augment_prediction_binary']
labels = {key: label_dic[key] for key in phenotype_subset}
odds_ratio_univariable = odds_ratios(phenotypes, diseases_unpack, labels,
                                     disease_list, covariates=[], instance=0, dont_scale=dont_scale)
plot_or_hr(odds_ratio_univariable, label_dic, disease_list, f'or_univariate_lighter_pretest')

hazard_ratio_univariable = hazard_ratios(phenotypes, diseases_unpack, labels,
                                         disease_list, covariates=[], instance=0, dont_scale=dont_scale)

plot_or_hr(hazard_ratio_univariable, label_dic, disease_list, f'hr_univariate_lighter_pretest', occ='incident')

# %%
# Multivariate
covariates = ['age', 'male']

phenotype_subset = ['LA_poisson_cleaned_max_binary',
                    'LA_Biplan_vol_max_binary',
                    'LA_poisson_cleaned_min_binary',
                    'LA_Biplan_vol_min_binary',
                    'LA_poisson_cleaned_max', 
                    'LA_Biplan_vol_max',
                    'LA_poisson_cleaned_min',
                    'LA_Biplan_vol_min',
                    ]

# phenotype_subset = ['LA_poisson_cleaned_max_binary',
#                     'LA_poisson_cleaned_min_binary',
#                     'LA_poisson_cleaned_max', 
#                     'LA_poisson_cleaned_min',
#                     ]
labels = {key: label_dic[key] for key in phenotype_subset}

odds_ratio_multivariable = odds_ratios(phenotypes, diseases_unpack, labels,
                                       disease_list, covariates=covariates, instance=2, dont_scale=dont_scale)
plot_or_hr(odds_ratio_multivariable, labels, disease_list, f'or_multivariate_petersen', occ='prevalent', horizontal_line_y=3.5)

hazard_ratio_multivariable = hazard_ratios(phenotypes, diseases_unpack, labels,
                                           disease_list, covariates=covariates, instance=2, dont_scale=dont_scale)
plot_or_hr(hazard_ratio_multivariable, labels, disease_list, f'hr_multivariate_petersen', occ='incident', horizontal_line_y=3.5)


# %%
fail_set = set(failed[0])
for tmp_set in failed[1:]:
    fail_set &= set(tmp_set)
# %%  
# Fail
def is_in_fail(data):
    if data in failed - success:
        return 1.0
    else:
        return 0.0
tmp_data = pd.read_csv('/home/pdiachil/ml/notebooks/mri/examine.csv')
tmp_data['failed'] = tmp_data['sample_id'].apply(is_in_fail)


# %%
# Clinical model
covariates = ['age', 'male', 'cholesterol', 'HDL', 'current_smoker',
              'systolic_bp', 'Diabetes_Type_2_prevalent', 'c_antihypertensive', 'c_lipidlowering']

phenotype_subset = ['bmi']
labels = {key: label_dic[key] for key in phenotype_subset}
hazard_ratio_clinical = hazard_ratios(phenotypes, diseases_unpack, labels,
                                      disease_list, covariates=covariates, instance=0, dont_scale=dont_scale)
plot_or_hr(hazard_ratio_clinical, labels, disease_list, f'hr_multivariate_pretest_clinical', occ='incident')

# %%
############## Resting ECGs ###############
# Read phenotype and covariates
# phenotypes = pd.read_csv('/home/pdiachil/ml/notebooks/genetics/pretest_covariates.csv')
phenotypes = pd.read_csv('/home/pdiachil/ml/notebooks/genetics/rest_covariates.csv')
# phenotypes = pd.read_csv('/home/pdiachil/ml/notebooks/genetics/overlap_covariates.csv')

# Median prediction
splits = [f'50_hrr_downsample_augment_split_{i}_prediction' for i in range(5)]
phenotypes['50_hrr_downsample_augment_prediction'] = np.median(phenotypes[splits], axis=1)

phenos_to_binarize = ['50_hrr_downsample_augment_prediction']
for pheno in phenos_to_binarize:
    phenotypes[f'{pheno}_binary'] = (phenotypes[pheno] < phenotypes[pheno].quantile(0.33)).apply(float)

label_dic = {
    '50_hrr_downsample_augment_prediction_binary': ['lowest tertile HRR$_{pred}$', ''],
    '50_hrr_downsample_augment_prediction': ['HRR$_{pred}$', 'beats'],
    'ventricular_rate': ['Rest HR', 'beats'],
    'age': ['Age', 'yrs'],
    'male': ['Male', ''],
    'nonwhite': ['Nonwhite', ''],
    'bmi': ['BMI', 'units'],
    'cholesterol': ['Cholesterol', 'mmol/L'],
    'HDL': ['HDL', 'mmol/L'],
    'current_smoker': ['Current smoker', ''],
    'diastolic_bp': ['Diastolic blood pressure', 'mmHg'],
    'systolic_bp': ['Systolic blood pressure', 'mmHg'],
    # 'gfr': ['eGFR', 'mL/min/1.73 m2'],
    # 'creatinine': ['Creatinine', 'umol/L'],
    'c_lipidlowering': ['Lipid lowering drugs', ''],
    'c_antihypertensive': ['Antihypertensive drugs', '']
}
dont_scale = ['male', 'nonwhite', 'current_smoker', 'c_lipidlowering', 'c_antihypertensive', '50_hrr_downsample_augment_prediction_binary']

# %%
# Read diseases and unpack
diseases = pd.read_csv('/home/pdiachil/ml/notebooks/genetics/bq_diseases.tsv', sep='\t')
diseases['censor_date'] = pd.to_datetime(diseases['censor_date'])

disease_list = [['Heart_Failure_V2', 'heart failure'],
                ['Diabetes_Type_2', 'type 2 diabetes'],
                ['composite_mi_cad_stroke_hf', 'CAD+stroke+HF+MI'],
                ]
diseases_unpack = unpack_disease(diseases, disease_list, phenotypes)


# %%
# Univariate
odds_ratio_univariable = odds_ratios(phenotypes, diseases_unpack, label_dic,
                                     disease_list, covariates=[], instance=2, dont_scale=dont_scale)
plot_or_hr(odds_ratio_univariable, label_dic, disease_list, f'or_univariate_rest')

hazard_ratio_univariable = hazard_ratios(phenotypes, diseases_unpack, label_dic,
                                         disease_list, covariates=[], instance=2, dont_scale=dont_scale)

plot_or_hr(hazard_ratio_univariable, label_dic, disease_list, f'hr_univariate_rest', occ='incident')

# Univariate lighter
phenotype_subset = ['50_hrr_downsample_augment_prediction', 'ventricular_rate',
                    '50_hrr_downsample_augment_prediction_binary']
labels = {key: label_dic[key] for key in phenotype_subset}
odds_ratio_univariable = odds_ratios(phenotypes, diseases_unpack, labels,
                                     disease_list, covariates=[], instance=2, dont_scale=dont_scale)
plot_or_hr(odds_ratio_univariable, label_dic, disease_list, f'or_univariate_lighter_rest')

hazard_ratio_univariable = hazard_ratios(phenotypes, diseases_unpack, labels,
                                         disease_list, covariates=[], instance=2, dont_scale=dont_scale)

plot_or_hr(hazard_ratio_univariable, label_dic, disease_list, f'hr_univariate_lighter_rest', occ='incident')


# %%
# %%
# Multivariate
covariates = ['bmi', 'age', 'male', 'cholesterol', 'HDL', 'current_smoker',
              'systolic_bp', 'Diabetes_Type_2_prevalent', 'c_antihypertensive', 'c_lipidlowering']

phenotype_subset = ['50_hrr_downsample_augment_prediction_binary',
                    '50_hrr_downsample_augment_prediction', 'ventricular_rate',
                    ]
labels = {key: label_dic[key] for key in phenotype_subset}

#odds_ratio_multivariable = odds_ratios(phenotypes, diseases_unpack, labels,
#                                       disease_list, covariates=covariates, instance=0, dont_scale=dont_scale)
#plot_or_hr(odds_ratio_multivariable, labels, disease_list, f'or_multivariate_pretest', occ='prevalent')

hazard_ratio_multivariable = hazard_ratios(phenotypes, diseases_unpack, labels,
                                           disease_list, covariates=covariates, instance=0, dont_scale=dont_scale)
plot_or_hr(hazard_ratio_multivariable, labels, disease_list, f'hr_multivariate_rest', occ='incident', horizontal_line_y=0.5)

# %%

# Linear regression on exercise ECGs
########### Pretest exercise ECGs ###########################
# Read phenotype and covariates
phenotypes = pd.read_csv('/home/pdiachil/ml/notebooks/genetics/pretest_covariates.csv')
# phenotypes = pd.read_csv('/home/pdiachil/ml/notebooks/genetics/rest_covariates.csv')
# phenotypes = pd.read_csv('/home/pdiachil/ml/notebooks/genetics/overlap_covariates.csv')
label_dic = {
    '50_hrr_actual': ['HRR50', 'beats'],
    '50_hrr_downsample_augment_prediction': ['HRR50-pretest', 'beats'],
    'resting_hr': ['HR-pretest', 'beats'],
    'age': ['Age', 'yrs'],
    'male': ['Male', ''],
    'nonwhite': ['Nonwhite', ''],
    'bmi': ['BMI', 'units'],
    'cholesterol': ['Cholesterol', 'mmol/L'],
    'HDL': ['HDL', 'mmol/L'],
    'current_smoker': ['Current smoker', ''],
    'diastolic_bp': ['Diastolic blood pressure', 'mmHg'],
    'systolic_bp': ['Systolic blood pressure', 'mmHg'],
    # 'gfr': ['eGFR', 'mL/min/1.73 m2'],
    # 'creatinine': ['Creatinine', 'umol/L'],
    'c_lipidlowering': ['Lipid lowering drugs', ''],
    'c_antihypertensive': ['Antihypertensive drugs', '']
}

dont_scale = ['male', 'nonwhite', 'current_smoker', 'c_lipidlowering', 'c_antihypertensive', '50_hrr_actual_binary', '50_hrr_downsample_augment_prediction_binary']

covariates = []
rsquared = {}
for covariate in ['resting_hr', 'bmi', 'age', 'male', 'cholesterol', 'HDL', 'current_smoker',
                  'diastolic_bp', 'systolic_bp', 'c_antihypertensive', 'c_lipidlowering', '50_hrr_downsample_augment_prediction']:
    covariates.append(covariate)

    res = random_forest_model(phenotypes, ['50_hrr_actual'], label_dic, covariates, dont_scale)
    rsquared[covariate] = {}
    rsquared[covariate]['mean'] = np.mean(res)
    rsquared[covariate]['std'] = np.std(res)

plot_rsquared_covariates(rsquared, label_dic)


# %%

# %%
