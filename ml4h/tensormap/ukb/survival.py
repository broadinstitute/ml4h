import h5py
import numpy as np
import datetime
from ml4h.TensorMap import TensorMap, Interpretation, str2date
from ml4h.defines import StorageType
from ml4h.metrics import weighted_crossentropy
from ml4h.tensormap.ukb.demographics import prevalent_tensor

DAYS_IN_5_YEARS = 365 * 5


def _survival_tensor(
    start_date_key: str,
    day_window: int,
    disease_name_override: str = None,
    incidence_only: bool = False,
    start_date_is_attribute: bool = False,
):
    def _survival_tensor_from_file(
        tm: TensorMap,
        hd5: h5py.File,
        dependents=None,
    ):
        if start_date_is_attribute:
            assess_date = datetime.datetime.utcfromtimestamp(hd5[start_date_key].attrs['date']).date()
        else:
            assess_date = str2date(str(hd5[start_date_key][0]))
        has_disease = 0  # Assume no disease if the tensor does not have the dataset

        if disease_name_override is not None:
            disease_name = disease_name_override
        else:
            disease_name = tm.name
        if disease_name in hd5['categorical']:
            has_disease = int(hd5['categorical'][disease_name][0])

        if disease_name + '_date' in hd5['dates']:
            censor_date = str2date(str(hd5['dates'][disease_name + '_date'][0]))
        elif 'phenotype_censor' in hd5['dates']:
            censor_date = str2date(str(hd5['dates/phenotype_censor'][0]))
        else:
            raise ValueError(f'No date found for survival {disease_name}')

        intervals = int(tm.shape[0] / 2)
        days_per_interval = day_window / intervals
        survival_then_censor = np.zeros(tm.shape, dtype=np.float32)
        for i, day_delta in enumerate(
                np.arange(0, day_window, days_per_interval),
        ):
            cur_date = assess_date + datetime.timedelta(days=day_delta)
            survival_then_censor[i] = float(cur_date < censor_date)
            survival_then_censor[intervals + i] = has_disease * float(
                censor_date <= cur_date < censor_date +
                datetime.timedelta(days=days_per_interval),
            )
            if i == 0 and censor_date <= cur_date:  # Handle prevalent diseases
                if incidence_only:
                    raise ValueError(f'{tm.name} ignores prior diagnoses.')
                survival_then_censor[intervals] = has_disease
        return survival_then_censor

    return _survival_tensor_from_file


def cox_tensor_from_file(
        start_date_key: str,
        incidence_only: bool = False,
        start_date_is_attribute: bool = False,
):
    def _cox_tensor_from_file(tm: TensorMap, hd5: h5py.File, dependents=None):
        if start_date_is_attribute:
            assess_date = datetime.datetime.utcfromtimestamp(hd5[start_date_key].attrs['date']).date()
        else:
            assess_date = str2date(str(hd5[start_date_key][0]))
        has_disease = 0  # Assume no disease if the tensor does not have the dataset
        if tm.name in hd5['categorical']:
            has_disease = int(hd5['categorical'][tm.name][0])

        if tm.name + '_date' in hd5['dates']:
            censor_date = str2date(str(hd5['dates'][tm.name + '_date'][0]))
        elif 'phenotype_censor' in hd5['dates']:
            censor_date = str2date(str(hd5['dates/phenotype_censor'][0]))
        else:
            raise ValueError(f'No date found for survival {tm.name}')

        if incidence_only and censor_date <= assess_date:
            raise ValueError(f'{tm.name} only considers incident diagnoses')

        tensor = np.zeros(tm.shape, dtype=np.float32)
        tensor[0] = has_disease
        tensor[1] = (censor_date - assess_date).days
        return tensor

    return _cox_tensor_from_file


enroll_cad_hazard = TensorMap(
    'coronary_artery_disease',
    Interpretation.SURVIVAL_CURVE,
    shape=(50,),
    days_window=DAYS_IN_5_YEARS,
    tensor_from_file=_survival_tensor('dates/enroll_date', DAYS_IN_5_YEARS),
)
enroll_hyp_hazard = TensorMap(
    'hypertension',
    Interpretation.SURVIVAL_CURVE,
    shape=(50,),
    days_window=DAYS_IN_5_YEARS,
    tensor_from_file=_survival_tensor('dates/enroll_date', DAYS_IN_5_YEARS),
)
enroll_afib_hazard = TensorMap(
    'atrial_fibrillation_or_flutter',
    Interpretation.SURVIVAL_CURVE,
    shape=(50,),
    days_window=DAYS_IN_5_YEARS,
    tensor_from_file=_survival_tensor('dates/enroll_date', DAYS_IN_5_YEARS),
)

enroll_chol_hazard = TensorMap(
    'hypercholesterolemia',
    Interpretation.SURVIVAL_CURVE,
    shape=(50,),
    days_window=DAYS_IN_5_YEARS,
    tensor_from_file=_survival_tensor('dates/enroll_date', DAYS_IN_5_YEARS),
)
enroll_diabetes2_hazard = TensorMap(
    'diabetes_type_2',
    Interpretation.SURVIVAL_CURVE,
    shape=(50,),
    days_window=DAYS_IN_5_YEARS,
    tensor_from_file=_survival_tensor('dates/enroll_date', DAYS_IN_5_YEARS),
)
enroll_diabetes2_hazard_incident = TensorMap(
    'diabetes_type_2',
    Interpretation.SURVIVAL_CURVE,
    shape=(50,),
    days_window=DAYS_IN_5_YEARS,
    tensor_from_file=_survival_tensor(
        'dates/enroll_date',
        DAYS_IN_5_YEARS,
        incidence_only=True,
    ),
)
enroll_hyp_hazard_5 = TensorMap(
    'hypertension',
    Interpretation.SURVIVAL_CURVE,
    shape=(50,),
    tensor_from_file=_survival_tensor('dates/enroll_date', DAYS_IN_5_YEARS),
)
enroll_hyp_hazard_5_incident = TensorMap(
    'hypertension',
    Interpretation.SURVIVAL_CURVE,
    shape=(50,),
    tensor_from_file=_survival_tensor(
        'dates/enroll_date',
        DAYS_IN_5_YEARS,
        incidence_only=True,
    ),
)
enroll_cad_hazard_5_incident = TensorMap(
    'coronary_artery_disease_soft',
    Interpretation.SURVIVAL_CURVE,
    shape=(50,),
    days_window=DAYS_IN_5_YEARS,
    tensor_from_file=_survival_tensor(
        'dates/enroll_date',
        DAYS_IN_5_YEARS,
        incidence_only=True,
    ),
)
enroll_cad_hazard_5 = TensorMap(
    'coronary_artery_disease_soft',
    Interpretation.SURVIVAL_CURVE,
    shape=(50,),
    days_window=DAYS_IN_5_YEARS,
    tensor_from_file=_survival_tensor('dates/enroll_date', DAYS_IN_5_YEARS),
)
enroll_mi_hazard_5 = TensorMap(
    'myocardial_infarction',
    Interpretation.SURVIVAL_CURVE,
    shape=(50,),
    days_window=DAYS_IN_5_YEARS,
    tensor_from_file=_survival_tensor('dates/enroll_date', DAYS_IN_5_YEARS),
)
enroll_mi_hazard_5_incident = TensorMap(
    'myocardial_infarction',
    Interpretation.SURVIVAL_CURVE,
    shape=(50,),
    days_window=DAYS_IN_5_YEARS,
    tensor_from_file=_survival_tensor(
        'dates/enroll_date',
        DAYS_IN_5_YEARS,
        incidence_only=True,
    ),
)

cox_mi = TensorMap(
    'myocardial_infarction',
    Interpretation.TIME_TO_EVENT,
    tensor_from_file=cox_tensor_from_file('dates/enroll_date'),
)
cox_mi_incident = TensorMap(
    'myocardial_infarction',
    Interpretation.TIME_TO_EVENT,
    tensor_from_file=cox_tensor_from_file(
        'dates/enroll_date', incidence_only=True,
    ),
)
cox_hyp = TensorMap(
    'hypertension',
    Interpretation.TIME_TO_EVENT,
    tensor_from_file=cox_tensor_from_file('dates/enroll_date'),
)
cox_hyp_incident = TensorMap(
    'hypertension',
    Interpretation.TIME_TO_EVENT,
    tensor_from_file=cox_tensor_from_file(
        'dates/enroll_date', incidence_only=True,
    ),
)
cox_cad = TensorMap(
    'coronary_artery_disease_soft',
    Interpretation.TIME_TO_EVENT,
    tensor_from_file=cox_tensor_from_file('dates/enroll_date'),
)
cox_cad_incident = TensorMap(
    'coronary_artery_disease_soft',
    Interpretation.TIME_TO_EVENT,
    tensor_from_file=cox_tensor_from_file(
        'dates/enroll_date', incidence_only=True,
    ),
)
cox_cad = TensorMap(
    'coronary_artery_disease_soft',
    Interpretation.TIME_TO_EVENT,
    tensor_from_file=cox_tensor_from_file('dates/enroll_date'),
)
cox_cad_incident = TensorMap(
    'coronary_artery_disease_soft',
    Interpretation.TIME_TO_EVENT,
    tensor_from_file=cox_tensor_from_file(
        'dates/enroll_date', incidence_only=True,
    ),
)

cox_afib_wrt_instance2 = TensorMap(
    'atrial_fibrillation_or_flutter',
    Interpretation.TIME_TO_EVENT,
    tensor_from_file=cox_tensor_from_file(
        'ukb_cardiac_mri/cine_segmented_lax_2ch/2/instance_0/', start_date_is_attribute=True,
    ),
)

survival_afib_wrt_instance2 = TensorMap(
    'atrial_fibrillation_or_flutter',
    Interpretation.SURVIVAL_CURVE,
    shape=(50,),
    days_window=DAYS_IN_5_YEARS,
    tensor_from_file=_survival_tensor(
        'ukb_cardiac_mri/cine_segmented_lax_2ch/2/instance_0/',
        DAYS_IN_5_YEARS, start_date_is_attribute=True,
    ),
)

mgb_afib_wrt_instance2 = TensorMap(
    'survival_curve_af',
    Interpretation.SURVIVAL_CURVE,
    shape=(50,),
    days_window=DAYS_IN_5_YEARS,
    tensor_from_file=_survival_tensor(
        'ukb_ecg_rest/ecg_rest_text/instance_2', DAYS_IN_5_YEARS,
        disease_name_override='atrial_fibrillation_or_flutter',
        start_date_is_attribute=True, incidence_only=True,
    ),
)
mgb_af_wrt_instance2 = TensorMap(
    'af_event',
    Interpretation.SURVIVAL_CURVE,
    shape=(50,),
    days_window=DAYS_IN_5_YEARS,
    tensor_from_file=_survival_tensor(
        'ukb_ecg_rest/ecg_rest_text/instance_2', DAYS_IN_5_YEARS,
        disease_name_override='atrial_fibrillation_or_flutter',
        start_date_is_attribute=True, incidence_only=True,
    ),
)
mgb_cad_wrt_instance2 = TensorMap(
    'cad_event',
    Interpretation.SURVIVAL_CURVE,
    shape=(50,),
    days_window=DAYS_IN_5_YEARS,
    tensor_from_file=_survival_tensor(
        'ukb_ecg_rest/ecg_rest_text/instance_2', DAYS_IN_5_YEARS,
        disease_name_override='coronary_artery_disease',
        start_date_is_attribute=True, incidence_only=True,
    ),
)
mgb_dm_wrt_instance2 = TensorMap(
    'dm_event',
    Interpretation.SURVIVAL_CURVE,
    shape=(50,),
    days_window=DAYS_IN_5_YEARS,
    tensor_from_file=_survival_tensor(
        'ukb_ecg_rest/ecg_rest_text/instance_2', DAYS_IN_5_YEARS,
        disease_name_override='diabetes_type_2',
        start_date_is_attribute=True, incidence_only=True,
    ),
)
mgb_mi_wrt_instance2 = TensorMap(
    'mi_event',
    Interpretation.SURVIVAL_CURVE,
    shape=(50,),
    days_window=DAYS_IN_5_YEARS,
    tensor_from_file=_survival_tensor(
        'ukb_ecg_rest/ecg_rest_text/instance_2', DAYS_IN_5_YEARS,
        disease_name_override='myocardial_infarction',
        start_date_is_attribute=True, incidence_only=True,
    ),
)

mgb_afib_as_hf_wrt_instance2 = TensorMap(
    'survival_curve_af',
    Interpretation.SURVIVAL_CURVE,
    shape=(50,),
    days_window=DAYS_IN_5_YEARS,
    tensor_from_file=_survival_tensor(
        'ukb_ecg_rest/ecg_rest_text/instance_2', DAYS_IN_5_YEARS,
        disease_name_override='heart_failure',
        start_date_is_attribute=True, incidence_only=True,
    ),
)

mgb_afib_as_stroke_wrt_instance2 = TensorMap(
    'survival_curve_af',
    Interpretation.SURVIVAL_CURVE,
    shape=(50,),
    days_window=DAYS_IN_5_YEARS,
    tensor_from_file=_survival_tensor(
        'ukb_ecg_rest/ecg_rest_text/instance_2', DAYS_IN_5_YEARS,
        disease_name_override='stroke',
        start_date_is_attribute=True, incidence_only=True,
    ),
)


prevalent_hf_wrt_instance2 = TensorMap(
    'heart_failure', Interpretation.CATEGORICAL, storage_type=StorageType.CATEGORICAL_FLAG,
    loss=weighted_crossentropy([1.0, 58], 'heart_failure'), path_prefix='categorical',
    channel_map={'no_heart_failure': 0, 'prevalent_heart_failure': 1},
    tensor_from_file=prevalent_tensor(
        'ukb_cardiac_mri/cine_segmented_lax_2ch/2/instance_0/',
        'dates/heart_failure_date',
        start_date_is_attribute=True,
    ),

)

prevalent_af_as_hf_wrt_instance2 = TensorMap(
    'heart_failure', Interpretation.CATEGORICAL, storage_type=StorageType.CATEGORICAL_FLAG,
    loss=weighted_crossentropy([1.0, 58], 'heart_failure'), path_prefix='categorical',
    channel_map={'no_atrial_fibrillation_or_flutter': 0, 'prevalent_atrial_fibrillation_or_flutter': 1},
    tensor_from_file=prevalent_tensor(
        'ukb_cardiac_mri/cine_segmented_lax_2ch/2/instance_0/',
        'dates/atrial_fibrillation_or_flutter_date',
        start_date_is_attribute=True,
    ),

)

prevalent_cad_wrt_instance2 = TensorMap(
    'coronary_artery_disease', Interpretation.CATEGORICAL, storage_type=StorageType.CATEGORICAL_FLAG,
    loss=weighted_crossentropy([1.0, 10], 'coronary_artery_disease'), path_prefix='categorical',
    channel_map={'no_coronary_artery_disease': 0, 'prevalent_coronary_artery_disease': 1},
    tensor_from_file=prevalent_tensor(
        'ukb_cardiac_mri/cine_segmented_lax_2ch/2/instance_0/',
        'dates/coronary_artery_disease_date',
        start_date_is_attribute=True,
    ),

)
