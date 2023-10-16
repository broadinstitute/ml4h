from ml4h.TensorMap import TensorMap, Interpretation


ecg_5000_std = TensorMap('ecg_5000_std', Interpretation.CONTINUOUS, shape=(5000, 12))

hypertension_icd_only = TensorMap(name='hypertension_icd_only', interpretation=Interpretation.CATEGORICAL,
                     channel_map={'no_hypertension_icd_only':0, 'hypertension_icd_only': 1})
hypertension_icd_bp = TensorMap(name='hypertension_icd_bp', interpretation=Interpretation.CATEGORICAL,
                     channel_map={'no_hypertension_icd_bp':0, 'hypertension_icd_bp': 1})
hypertension_icd_bp_med = TensorMap(name='hypertension_icd_bp_med', interpretation=Interpretation.CATEGORICAL,
                     channel_map={'no_hypertension_icd_bp_med':0, 'hypertension_icd_bp_med': 1})

lvef_tmap = TensorMap(name='LVEF', interpretation=Interpretation.CONTINUOUS, channel_map={'LVEF': 0})

age_tmap = TensorMap(name='age_in_days', interpretation=Interpretation.CONTINUOUS, channel_map={'age_in_days': 0})
sex_tmap = TensorMap(name='sex', interpretation=Interpretation.CATEGORICAL, channel_map={'Female': 0, 'Male':1})


