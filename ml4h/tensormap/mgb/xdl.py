from ml4h.TensorMap import TensorMap, Interpretation

ecg_5000_std = TensorMap('ecg_5000_std', Interpretation.CONTINUOUS, shape=(5000, 12))

hypertension_icd_only = TensorMap(name='hypertension_icd_only', interpretation=Interpretation.CATEGORICAL,
                                  channel_map={'no_hypertension_icd_only': 0, 'hypertension_icd_only': 1})
hypertension_icd_bp = TensorMap(name='hypertension_icd_bp', interpretation=Interpretation.CATEGORICAL,
                                channel_map={'no_hypertension_icd_bp': 0, 'hypertension_icd_bp': 1})
hypertension_icd_bp_med = TensorMap(name='hypertension_icd_bp_med', interpretation=Interpretation.CATEGORICAL,
                                    channel_map={'no_hypertension_icd_bp_med': 0, 'hypertension_icd_bp_med': 1})

lvef = TensorMap(name='LVEF', interpretation=Interpretation.CONTINUOUS, channel_map={'LVEF': 0})

age = TensorMap(name='age_in_days', interpretation=Interpretation.CONTINUOUS, channel_map={'age_in_days': 0})
sex = TensorMap(name='sex', interpretation=Interpretation.CATEGORICAL, channel_map={'Female': 0, 'Male': 1})

cad = TensorMap(name='cad', interpretation=Interpretation.CATEGORICAL, channel_map={'no_cad': 0, 'cad': 1})
dm = TensorMap(name='dm', interpretation=Interpretation.CATEGORICAL, channel_map={'no_dm': 0, 'dm': 1})
hypercholesterolemia = TensorMap(name='hypercholesterolemia', interpretation=Interpretation.CATEGORICAL,
                                      channel_map={'no_hypercholesterolemia': 0, 'hypercholesterolemia': 1})
