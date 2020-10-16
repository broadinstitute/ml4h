import numpy as np
import pandas as pd
import statsmodels.api as sm
from matplotlib import pyplot as plt
from sklearn.metrics import roc_auc_score, r2_score, roc_curve, f1_score, average_precision_score
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression

def unpack_disease(diseases, disease_list, phenotypes):

    diseases_unpack = pd.DataFrame()
    diseases_unpack['sample_id'] = np.unique(np.hstack([diseases['sample_id'], phenotypes['sample_id']]))
    for disease, disease_label in disease_list:
        tmp_diseases = diseases[(diseases['disease']==disease) &\
                                (diseases['incident_disease'] > 0.5)]
        tmp_diseases_unpack = diseases_unpack.merge(tmp_diseases[['sample_id', 'incident_disease', 'censor_date']], how='left', on='sample_id')
        diseases_unpack[f'{disease}_incident'] = tmp_diseases_unpack['incident_disease']
        incident_cases = np.logical_not(diseases_unpack[f'{disease}_incident'].isna())
        diseases_unpack.loc[incident_cases, f'{disease}_censor_date'] = tmp_diseases_unpack.loc[incident_cases, 'censor_date']
        tmp_diseases = diseases[(diseases['disease']==disease) &\
                                (diseases['prevalent_disease'] > 0.5)]
        tmp_diseases_unpack = diseases_unpack.merge(tmp_diseases[['sample_id', 'prevalent_disease', 'censor_date']], how='left', on='sample_id')
        diseases_unpack[f'{disease}_prevalent'] = tmp_diseases_unpack['prevalent_disease']
        prevalent_cases = np.logical_not(diseases_unpack[f'{disease}_prevalent'].isna())
        diseases_unpack.loc[prevalent_cases, f'{disease}_censor_date'] = tmp_diseases_unpack.loc[prevalent_cases, 'censor_date']
        diseases_unpack.loc[diseases_unpack[f'{disease}_censor_date'].isna(), f'{disease}_censor_date'] = pd.to_datetime('2020-03-31')

    # If NaN, disease is absent
    diseases_unpack = diseases_unpack.fillna(0)
    return diseases_unpack

def odds_ratios(
    phenotypes, diseases_unpack, labels, disease_labels,
    covariates, instance, dont_scale,
):
    or_multi_dic = {}
    for pheno in labels:
        if pheno in covariates:
            continue
        or_multi_dic[pheno] = {}
        tmp_pheno = phenotypes[['sample_id', pheno, f'instance{instance}_date'] + covariates]
        tmp_pheno[f'instance{instance}_date'] = pd.to_datetime(tmp_pheno[f'instance{instance}_date'])
        for disease, disease_label in disease_labels:
            or_multi_dic[pheno][f'{disease}_prevalent'] = {}
            tmp_data = tmp_pheno.merge(diseases_unpack[['sample_id', f'{disease}_censor_date', f'{disease}_prevalent', f'{disease}_incident']], on='sample_id')
            tmp_data[f'{disease}_prevalent'] = (tmp_data[f'instance{instance}_date'] >= tmp_data[f'{disease}_censor_date']).apply(float)
            if pheno in dont_scale:
                std = ''
            else:
                std = np.std(tmp_data[pheno].values)
                tmp_data[pheno] = (tmp_data[pheno].values - np.mean(tmp_data[pheno].values))/std
                std = f', {std:.1f}'
            covariates_scale = [covariate for covariate in covariates if covariate not in dont_scale]
            tmp_data[covariates_scale] = (
                tmp_data[covariates_scale].values \
                                          - np.mean(tmp_data[covariates_scale].values, axis=0)
            )/\
                                          np.std(tmp_data[covariates_scale].values, axis=0)
            tmp_data['intercept'] = 1.0
            res = sm.Logit(tmp_data[f'{disease}_prevalent'], tmp_data[[pheno, 'intercept']+covariates]).fit(disp=False)
            res_predict = res.predict(tmp_data[[pheno, 'intercept']+covariates])
            or_multi_dic[pheno][f'{disease}_prevalent']['OR'] = np.exp(res.params[0])
            or_multi_dic[pheno][f'{disease}_prevalent']['CI'] = np.exp(res.conf_int().values[0])
            or_multi_dic[pheno][f'{disease}_prevalent']['p'] = res.pvalues[pheno]
            or_multi_dic[pheno][f'{disease}_prevalent']['n'] = np.sum(tmp_data[f'{disease}_prevalent'])
            or_multi_dic[pheno][f'{disease}_prevalent']['ntot'] = len(tmp_data)
            or_multi_dic[pheno][f'{disease}_prevalent']['std'] = std
            or_multi_dic[pheno][f'{disease}_prevalent']['auc'] = roc_auc_score(tmp_data[f'{disease}_prevalent'], res_predict)

    return or_multi_dic


def hazard_ratios(
    phenotypes, diseases_unpack, labels, disease_labels,
    covariates, instance, dont_scale,
):
    diabetes_is_covariate = 'Diabetes_Type_2_prevalent' in covariates
    if diabetes_is_covariate:
        covariates.remove('Diabetes_Type_2_prevalent')
    hr_multi_dic = {}
    for pheno in labels:
        if pheno in covariates:
            continue
        hr_multi_dic[pheno] = {}
        tmp_pheno = phenotypes[['sample_id', pheno, f'instance{instance}_date'] + covariates]
        tmp_pheno[f'instance{instance}_date'] = pd.to_datetime(tmp_pheno[f'instance{instance}_date'])
        for disease, disease_label in disease_labels:
            hr_multi_dic[pheno][f'{disease}_incident'] = {}
            tmp_data = tmp_pheno.merge(diseases_unpack[['sample_id', f'{disease}_censor_date', f'{disease}_prevalent', f'{disease}_incident']], on='sample_id')
            if diabetes_is_covariate and disease != 'Diabetes_Type_2':
                tmp_data = tmp_data.merge(diseases_unpack[['sample_id', f'Diabetes_Type_2_censor_date', f'Diabetes_Type_2_prevalent']], on='sample_id')
                tmp_data[f'Diabetes_Type_2_prevalent'] = (tmp_data[f'instance{instance}_date'] >= tmp_data[f'Diabetes_Type_2_censor_date']).apply(float)

            tmp_data[f'{disease}_prevalent'] = (tmp_data[f'instance{instance}_date'] >= tmp_data[f'{disease}_censor_date']).apply(float)
            tmp_data[f'{disease}_incident'] = (
                np.logical_and(
                    (tmp_data[f'instance{instance}_date'] < tmp_data[f'{disease}_censor_date']),
                    (tmp_data[f'{disease}_censor_date'] < pd.to_datetime('2020-03-31')),
                )
            ).apply(float)
            tmp_data = tmp_data[tmp_data[f'{disease}_prevalent']<0.5]

            if pheno in dont_scale:
                std = ''
            else:
                std = np.std(tmp_data[pheno].values)
                tmp_data[pheno] = (tmp_data[pheno].values - np.mean(tmp_data[pheno].values))/std
                std = f', {std:.1f}'
            covariates_scale = [covariate for covariate in covariates if covariate not in dont_scale]
            tmp_data[covariates_scale] = (
                tmp_data[covariates_scale].values \
                                         - np.mean(tmp_data[covariates_scale].values, axis=0)
            )/\
                                         np.std(tmp_data[covariates_scale].values, axis=0)
            tmp_data['intercept'] = 1.0
            tmp_data['futime'] = (tmp_data[f'{disease}_censor_date']-tmp_data[f'instance{instance}_date']).dt.days
            tmp_data['entry'] = 0.0
            tmp_data = tmp_data[tmp_data['futime']>0]
            regression_covariates = [covariate for covariate in covariates]
            if diabetes_is_covariate and disease != 'Diabetes_Type_2':
                regression_covariates += ['Diabetes_Type_2_prevalent']
            tmp_data.to_csv('/home/pdiachil/ml/notebooks/mri/examine.csv')
            res = sm.PHReg(
                tmp_data['futime'], tmp_data[[pheno]+regression_covariates],
                tmp_data[f'{disease}_incident'], tmp_data['entry'],
            ).fit()
            res_predict = res.predict()
            hr_multi_dic[pheno][f'{disease}_incident']['HR'] = np.exp(res.params[0])
            hr_multi_dic[pheno][f'{disease}_incident']['CI'] = np.exp(res.conf_int()[0])
            hr_multi_dic[pheno][f'{disease}_incident']['p'] = res.pvalues[0]
            hr_multi_dic[pheno][f'{disease}_incident']['n'] = np.sum(tmp_data[f'{disease}_incident'])
            hr_multi_dic[pheno][f'{disease}_incident']['ntot'] = len(tmp_data)
            hr_multi_dic[pheno][f'{disease}_incident']['std'] = std
            hr_multi_dic[pheno][f'{disease}_incident']['auc'] = roc_auc_score(tmp_data[[f'{disease}_incident']], res_predict.predicted_values)
            hr_multi_dic[pheno][f'{disease}_incident']['prauc'] = average_precision_score(tmp_data[[f'{disease}_incident']], res_predict.predicted_values)
    return hr_multi_dic


def plot_or_hr(or_dic, label_dic, disease_list, suffix, occ='prevalent', horizontal_line_y=None):
    ratio_type = 'OR' if 'prevalent' in occ else 'HR'
    ratio_label = 'Odds' if 'prevalent' in occ else 'Hazard'
    for dis, dis_label in disease_list:
        ors = []
        cis_minus = []
        cis_plus = []
        labels = []
        for pheno in or_dic:
            ors.append(np.exp(np.log(or_dic[pheno][f'{dis}_{occ}'][ratio_type])))
            cis_minus.append(np.exp(np.log(or_dic[pheno][f'{dis}_{occ}'][ratio_type]-or_dic[pheno][f'{dis}_{occ}']['CI'][0])))
            cis_plus.append(np.exp(np.log(or_dic[pheno][f'{dis}_{occ}']['CI'][1]-or_dic[pheno][f'{dis}_{occ}'][ratio_type])))
            label = f'{label_dic[pheno][0]}{or_dic[pheno][dis+"_"+occ]["std"]} {label_dic[pheno][1]}'
            label += f'  {or_dic[pheno][dis+"_"+occ]["auc"]:.2f}'
            labels.append(label)

            if or_dic[pheno][f'{dis}_{occ}']['p'] < 0.05:
                labels[-1] += '*'

        f, ax = plt.subplots()
        f.set_size_inches(6.5, 4)
        ax.errorbar(ors, np.arange(len(ors)), xerr=(cis_minus, cis_plus), marker='o', linestyle='', color='black')
        ax.plot([1.0, 1.0], [-1.0, len(ors)], 'k-')
        if horizontal_line_y:
            ax.plot([-20.0, 20.0], [horizontal_line_y, horizontal_line_y], 'k:')
        ax.set_yticks(np.arange(len(ors)))
        ax.set_yticklabels(labels)
        ax.set_xscale('log', basex=np.exp(1))
        ax.set_xticks([0.25, 0.5, 1.0, 2.0, 4.0, 8.0])
        ax.set_xticklabels(map(str, [0.25, 0.5, 1.0, 2.0, 4.0, 8.0]))
        ax.set_xlabel(f'{ratio_label} ratio (per 1-SD increase of continuous variables)')
        ax.set_title(f'{occ} {dis_label}\n n$_+$ = {int(or_dic[pheno][dis+"_"+occ]["n"])} / {int(or_dic[pheno][dis+"_"+occ]["ntot"])}')
        ax.set_ylim([-1.0, len(ors)])
        ax.set_xlim([min(1.0, min(ors)*0.8), max(4.0, max(ors)*1.5)])
        plt.tight_layout()
        f.savefig(f'{dis}_{occ}_{suffix}.png', dpi=500)


def regression_model(
    phenotypes, pheno_list, labels,
    covariates, dont_scale,
):

    r_multi_dic = {}
    for pheno in pheno_list:
        if pheno in covariates:
            continue
        tmp_pheno = phenotypes[['sample_id', pheno] + covariates]
        tmp_data = tmp_pheno
        if pheno in dont_scale:
            std = ''
        else:
            std = np.std(tmp_data[pheno].values)
            tmp_data[pheno] = (tmp_data[pheno].values - np.mean(tmp_data[pheno].values))/std
            std = f', {std:.1f}'
        covariates_scale = [covariate for covariate in covariates if covariate not in dont_scale]
        tmp_data[covariates_scale] = (
            tmp_data[covariates_scale].values \
                                      - np.mean(tmp_data[covariates_scale].values, axis=0)
        )/\
                                      np.std(tmp_data[covariates_scale].values, axis=0)
        tmp_data['intercept'] = 1.0
        res = sm.OLS(tmp_data[pheno], tmp_data[['intercept']+covariates]).fit(disp=False)

    return res


def logistic_regression_model(
    phenotypes, pheno_list, labels,
    covariates, dont_scale,
):
    r_multi_dic = {}
    for pheno in pheno_list:
        if pheno in covariates:
            continue
        tmp_pheno = phenotypes[['sample_id', pheno] + covariates]
        tmp_data = tmp_pheno
        if pheno in dont_scale:
            std = ''
        else:
            std = np.std(tmp_data[pheno].values)
            tmp_data[pheno] = (tmp_data[pheno].values - np.mean(tmp_data[pheno].values))/std
            std = f', {std:.1f}'
        covariates_scale = [covariate for covariate in covariates if covariate not in dont_scale]
        tmp_data[covariates_scale] = (
            tmp_data[covariates_scale].values \
                                      - np.mean(tmp_data[covariates_scale].values, axis=0)
        )/\
                                      np.std(tmp_data[covariates_scale].values, axis=0)
        kf = KFold(n_splits=5)
        r2s = {}
        r2s['R2'] = []
        r2s['auc'] = []
        r2s['prauc'] = []
        r2s['fpr'] = []
        r2s['tpr'] = []

        mean_fpr = np.linspace(0, 1, 1000)
        for train_index, test_index in kf.split(tmp_data):
            rf = LogisticRegression(random_state=0)
            rf.fit(tmp_data[covariates].values[train_index], tmp_data[pheno].values[train_index])
            predict = rf.predict(tmp_data[covariates].values[test_index])
            predict_score = rf.predict_proba(tmp_data[covariates].values[test_index])[:, 1]
            r2s['R2'].append(r2_score(tmp_data[pheno].values[test_index], predict))
            r2s['auc'].append(roc_auc_score(tmp_data[pheno].values[test_index], predict_score))
            r2s['prauc'].append(average_precision_score(tmp_data[pheno].values[test_index], predict_score))
            fpr, tpr, thresh = roc_curve(tmp_data[pheno].values[test_index], predict_score)
            interp_tpr = np.interp(mean_fpr, fpr, tpr)
            r2s['tpr'].append(interp_tpr)
            r2s['fpr'].append(mean_fpr)
    return r2s

def random_forest_model(
    phenotypes, pheno_list, labels,
    covariates, dont_scale,
):

    r_multi_dic = {}
    for pheno in pheno_list:
        if pheno in covariates:
            continue
        tmp_pheno = phenotypes[['sample_id', pheno] + covariates]
        tmp_data = tmp_pheno
        if pheno in dont_scale:
            std = ''
        else:
            std = np.std(tmp_data[pheno].values)
            tmp_data[pheno] = (tmp_data[pheno].values - np.mean(tmp_data[pheno].values))/std
            std = f', {std:.1f}'
        covariates_scale = [covariate for covariate in covariates if covariate not in dont_scale]
        tmp_data[covariates_scale] = (
            tmp_data[covariates_scale].values \
                                      - np.mean(tmp_data[covariates_scale].values, axis=0)
        )/\
                                      np.std(tmp_data[covariates_scale].values, axis=0)
        kf = KFold(n_splits=5)
        r2s = []
        for train_index, test_index in kf.split(tmp_data):
            rf = RandomForestRegressor(max_depth=5)
            rf.fit(tmp_data[covariates].values[train_index], tmp_data[pheno].values[train_index])
            predict = rf.predict(tmp_data[covariates].values[test_index])
            r2s.append(r2_score(tmp_data[pheno].values[test_index], predict))

    return r2s

def plot_rsquared_covariates(rsquared, label_dic, fname, xlabel=True, height=3, horizontal_line_y=None, start_plus=0):
    f, ax = plt.subplots()
    f.set_size_inches(5, height)
    yticks = []
    for i, rr in enumerate(rsquared):
        ycoor = len(rsquared)-i-1
        if ycoor > horizontal_line_y:
            ycoor += 0.5
        yticks.append(ycoor)
        if i == 0:
            ax.barh([ycoor], rsquared[rr]['mean'], xerr=rsquared[rr]['std'], color='black')
        elif i == 1:
            ax.barh([ycoor], rsquared[rr]['mean'], xerr=rsquared[rr]['std']*4, color='white', edgecolor='black')
        else:
            ax.barh([ycoor], rsquared[rr]['mean'], xerr=rsquared[rr]['std'], color='gray', edgecolor='black')
    ticklabels = []
    for i, covariate in enumerate(rsquared):
        prefix = '' if i <= start_plus else '+ '
        ticklabels.append(prefix+label_dic[covariate][0])
    ax.set_yticks(yticks)
    ax.set_yticklabels(ticklabels)
    ax.set_xlabel('R$^2$')
    ax.set_xlim([0.0, 0.32])
    ax.set_xticks([0.0, 0.1, 0.2, 0.3])
    if not xlabel:
        ax.set_xticks([])
        ax.set_xlabel('')
    if horizontal_line_y:
        ax.plot([0.0, 0.32], [horizontal_line_y, horizontal_line_y], 'k--')
    plt.tight_layout()
    f.savefig(f'{fname}.png', dpi=500)
