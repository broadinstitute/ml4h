# %%
import pandas as pd
%load_ext google.cloud.bigquery

# %%
petersen = pd.read_csv('/home/pdiachil/returned_lv_mass.tsv', sep='\t')
petersen = petersen.dropna()
# %%
from scipy.interpolate import interp1d
from scipy.optimize import minimize
import numpy as np

def err_variable_elastance(t_lvpmax, time, lv_vol, lv_sp, E_min, tau1, tau2, n1, n2, V0, plot=False):
    if (t_lvpmax < np.min(time)) or (t_lvpmax>np.max(time)):
        return 1000.0
    lv_vol_int = interp1d(time, lv_vol)
    E_max = lv_sp / (lv_vol_int(t_lvpmax)-V0)

    g1 = (time/tau1) ** n1
    g2 = (time/tau2)**n2

    k = (E_max - E_min)/np.max((g1/(1.0+g1)/(1.0+g2)))
    E = k * g1 / (1.0+g1) / (1.0+g2) + E_min

    if plot:
        argt_lvpmax = np.argmin(np.abs(time-t_lvpmax))
        lv_press = E*lv_vol
        f, ax = plt.subplots(2, 1)
        ax[0].plot(time, lv_press)
        ax[0].plot([0, time[-1]], [lv_sp, lv_sp])
        ax[0].plot(time[argt_lvpmax], lv_press[argt_lvpmax], '*')

        ax[1].plot(time, lv_vol)

        f, ax = plt.subplots()
        ax.plot(lv_vol, lv_press)

        return lv_vol, lv_press
    else:
        return ((lv_sp - np.max(E*lv_vol))/lv_sp)**2.0


def inverse_variable_elastance(heart_rate, lv_vol, lv_sp, lv_dp, n1=1.63, n2=24.2, tau1=0.113, tau2=1.035, V0=0):
    t_cycle = 60.0/heart_rate
    time = np.linspace(0.0, 60.0/heart_rate, 50)
    t_ed = 0.0
    t_es = time[np.argmin(lv_vol)]
    lv_dv = lv_vol[0]

    tau1 = tau1 * t_cycle
    tau2 = tau2 * t_es
    E_min = lv_dp / (lv_dv - V0)

    t0 = 0.10
    t_lvpmax = minimize(err_variable_elastance, t0, method='Nelder-Mead', args=(time, lv_vol, lv_sp, E_min, tau1, tau2, n1, n2, V0, False))
    lv_vol, lv_press = err_variable_elastance(t_lvpmax.x, time, lv_vol, lv_sp, E_min, tau1, tau2, n1, n2, V0, True)
    return time, lv_vol, lv_press
# %%
sample_id = 5042648 #hfref
sample_id = 2002767 #htn
param_dic = {'sample_id': sample_id}
lv = pd.read_csv(f'/home/pdiachil/ml/notebooks/mri/ventricles_processed_{sample_id}_{sample_id+1}.csv')
cols = [f'LV_poisson_{d}' for d in range(50)]

# %%
%%bigquery covariates --params $param_dic
select sample_id, FieldID, instance, array_idx, value from `ukbb7089_201910.phenotype`
where sample_id = @sample_id
and instance = 2

# %%
import matplotlib.pyplot as plt
covariate_dic = {'heart_rate': 22426, 'central_systolic_bp': 12677, 'SBP': 4080, 'DBP': 4079}
covariate_values = {}
for covariate in covariate_dic:
    val = covariates[covariates['FieldID']==covariate_dic[covariate]]['value'].values[0]
    covariate_values[covariate] = val

lv_sp = 0.67*int(covariate_values['SBP']) + 0.33*int(covariate_values['DBP'])
t_htn, lv_vol_htn, lv_press_htn = inverse_variable_elastance(float(covariate_values['heart_rate']),
                                                              lv[cols].values[0], lv_sp, 10.0)
# %%
sample_id = 5042648 #hfref
# sample_id = 2002767 #healthy
param_dic = {'sample_id': sample_id}
lv = pd.read_csv(f'/home/pdiachil/ml/notebooks/mri/ventricles_processed_{sample_id}_{sample_id+1}.csv')
cols = [f'LV_poisson_{d}' for d in range(50)]

#%%
%%bigquery covariates --params $param_dic
select sample_id, FieldID, instance, array_idx, value from `ukbb7089_201910.phenotype`
where sample_id = @sample_id
and instance = 2

# %%
import matplotlib.pyplot as plt
covariate_dic = {'heart_rate': 22426, 'central_systolic_bp': 12677, 'SBP': 4080, 'DBP': 4079}
covariate_values = {}
for covariate in covariate_dic:
    val = covariates[covariates['FieldID']==covariate_dic[covariate]]['value'].values[0]
    covariate_values[covariate] = val

lv_sp = 0.67*int(covariate_values['SBP']) + 0.33*int(covariate_values['DBP'])
t_hfref, lv_vol_hfref, lv_press_hfref, = inverse_variable_elastance(float(covariate_values['heart_rate']),
                                                           lv[cols].values[0], lv_sp, 10.0)
# %%
sample_id = 5042648 #hfref
# sample_id = 2002767 #htn
sample_id = 4941570 # healthy
param_dic = {'sample_id': sample_id}
lv = pd.read_csv(f'/home/pdiachil/ml/notebooks/mri/ventricles_processed_{sample_id}_{sample_id+1}.csv')
cols = [f'LV_poisson_{d}' for d in range(50)]

#%%
%%bigquery covariates --params $param_dic
select sample_id, FieldID, instance, array_idx, value from `ukbb7089_201910.phenotype`
where sample_id = @sample_id
and instance = 2

# %%
import matplotlib.pyplot as plt
covariate_dic = {'heart_rate': 22426, 'central_systolic_bp': 12677, 'SBP': 4080, 'DBP': 4079}
covariate_values = {}
for covariate in covariate_dic:
    val = covariates[covariates['FieldID']==covariate_dic[covariate]]['value'].values[0]
    covariate_values[covariate] = val

lv_sp = 0.67*int(covariate_values['SBP']) + 0.33*int(covariate_values['DBP'])
t_healthy, lv_vol_healthy, lv_press_healthy, = inverse_variable_elastance(float(covariate_values['heart_rate']),
                                                           lv[cols].values[0], lv_sp, 10.0)
# %%
f, ax = plt.subplots(1, 2)
f.set_size_inches(8, 3.0)
ax[1].plot(lv_vol_htn, lv_press_htn, label='hypertension', color=[0.4, 0.4, 0.4], linewidth=3)
ax[1].plot(lv_vol_hfref, lv_press_hfref, label='low EF', color=[0.8, 0.8, 0.8], linewidth=3)
ax[1].plot(lv_vol_healthy, lv_press_healthy, label='healthy', color='black', linewidth=3)
ax[1].set_ylim([0.0, 150.0])

l1 = ax[0].plot(t_healthy, lv_press_healthy, label='healthy', color='black', linewidth=3)
l2 = ax[0].plot(t_htn, lv_press_htn, label='hypertension', color=[0.4, 0.4, 0.4], linewidth=3)
l3 = ax[0].plot(t_hfref, lv_press_hfref, label='low EF', color=[0.8, 0.8, 0.8], linewidth=3)


ax[0].set_ylabel('LV pressure (mmHg)')
ax[0].set_xlabel('Time (s)')
ax[0].set_ylim([0.0, 150.0])
ax[1].set_yticklabels([])
ax[1].set_xlabel('LV volume (ml)')
lgd = f.legend([l1[0], l2[0], l3[0]],
         labels=['healthy', 'hypertension', 'low EF'],
         loc='upper center', ncol=3, bbox_to_anchor=(0.5, 0.55, 0., 0.5))
plt.tight_layout()
f.savefig('pvloops.png', dpi=500, bbox_extra_artists=(lgd,), bbox_inches='tight')

# %%
f, ax = plt.subplots(1, 3)
f.set_size_inches(10, 3)
ax[0].plot(t_healthy, lv_vol_healthy, color='black', linewidth=3)
ax[0].set_xlim([t_healthy[0], t_healthy[-1]])
ax[1].plot(t_htn, lv_vol_htn, color=[0.4, 0.4, 0.4], linewidth=3)
ax[1].set_xlim([t_htn[0], t_htn[-1]])
ax[2].plot(t_hfref, lv_vol_hfref, color=[0.8, 0.8, 0.8], linewidth=3)
ax[2].set_xlim([t_hfref[0], t_hfref[-1]])
ax[0].set_ylim([80.0, 220.0])
ax[1].set_ylim([80.0, 220.0])
ax[2].set_ylim([80.0, 220.0])
ax[1].set_yticklabels([])
ax[2].set_yticklabels([])
ax[0].set_ylabel('LV volume (ml)')
ax[0].set_xlabel('Time (s)')
ax[1].set_xlabel('Time (s)')
ax[2].set_xlabel('Time (s)')
plt.tight_layout()
f.savefig('lvedvs.png', dpi=500)
# %%
