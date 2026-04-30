import h5py
import numpy as np
from datetime import datetime
import logging
from typing import List, Tuple

from ml4h.normalizer import Standardize, ZeroMeanStd1
from ml4h.tensormap.general import tensor_path
from ml4h.TensorMap import TensorMap, Interpretation, str2date, make_range_validator
from ml4h.defines import StorageType


def is_genetic_man(hd5):
    return 'Genetic-sex_Male_0_0' in hd5['categorical']


def is_genetic_woman(hd5):
    return 'Genetic-sex_Female_0_0' in hd5['categorical']


def age_in_years_tensor(
    date_key,
    birth_key='continuous/34_Year-of-birth_0_0',
    population_normalize=False,
):
    def age_at_tensor_from_file(
        tm: TensorMap,
        hd5: h5py.File,
        dependents=None,
    ):
        try:
            age = np.array([hd5['ecg/latest/patient_info/Age'][()]])
        except:
            logging.info('could not get age')
            raise KeyError('cold not')
        # age = age.astype("float")

        return age
        # return tm.normalize_and_validate(np.array([assess_date.year-birth_year]))

    return age_at_tensor_from_file


def prevalent_incident_tensor(start_date_key, event_date_key):
    def _prevalent_incident_tensor_from_file(
        tm: TensorMap,
        hd5: h5py.File,
        dependents=None,
    ):
        index = 0
        categorical_data = np.zeros(tm.shape, dtype=np.float32)
        if tm.hd5_key_guess() in hd5:
            data = tm.hd5_first_dataset_in_group(hd5, tm.hd5_key_guess())
            if tm.storage_type == StorageType.CATEGORICAL_INDEX or tm.storage_type == StorageType.CATEGORICAL_FLAG:
                index = int(data[0])
                categorical_data[index] = 1.0
            else:
                categorical_data = np.array(data)
        elif tm.storage_type == StorageType.CATEGORICAL_FLAG:
            categorical_data[index] = 1.0
        else:
            raise ValueError(
                f"No HD5 Key at prefix {tm.path_prefix} found for tensor map: {tm.name}.",
            )

        if index != 0:
            if event_date_key in hd5 and start_date_key in hd5:
                disease_date = str2date(str(hd5[event_date_key][0]))
                assess_date = str2date(str(hd5[start_date_key][0]))
            else:
                raise ValueError(f"No date found for tensor map: {tm.name}.")
            index = 1 if disease_date < assess_date else 2
        categorical_data[index] = 1.0
        return categorical_data

    return _prevalent_incident_tensor_from_file


def prevalent_tensor(start_date_key: str, event_date_key: str, start_date_is_attribute: bool = False):
    def _prevalent_tensor_from_file(
        tm: TensorMap,
        hd5: h5py.File,
        dependents=None,
    ):
        index = 0
        categorical_data = np.zeros(tm.shape, dtype=np.float32)
        if tm.hd5_key_guess() in hd5:
            data = tm.hd5_first_dataset_in_group(hd5, tm.hd5_key_guess())
            if tm.storage_type == StorageType.CATEGORICAL_INDEX or tm.storage_type == StorageType.CATEGORICAL_FLAG:
                index = int(data[0])
                categorical_data[index] = 1.0
            else:
                categorical_data = np.array(data)
        elif tm.storage_type == StorageType.CATEGORICAL_FLAG:
            categorical_data[index] = 1.0
        else:
            raise ValueError(
                f"No HD5 Key at prefix {tm.path_prefix} found for tensor map: {tm.name}.",
            )

        if index != 0:
            if event_date_key in hd5 and start_date_key in hd5:
                disease_date = str2date(str(hd5[event_date_key][0]))
                if start_date_is_attribute:
                    assess_date = datetime.utcfromtimestamp(hd5[start_date_key].attrs['date']).date()
                else:
                    assess_date = str2date(str(hd5[start_date_key][0]))
            else:
                raise ValueError(f"No date found for tensor map: {tm.name}.")
            index = 1 if disease_date < assess_date else 0
        categorical_data[index] = 1.0
        return categorical_data
    return _prevalent_tensor_from_file


def preprocess_with_function(fxn, hd5_key=None):
    def preprocess_tensor_from_file(tm, hd5, dependents={}):
        missing = True
        continuous_data = np.zeros(tm.shape, dtype=np.float32)
        my_key = tm.hd5_key_guess() if hd5_key is None else hd5_key
        if my_key in hd5:
            missing = False
            continuous_data[0] = tm.hd5_first_dataset_in_group(hd5, my_key)[0]
        if missing and tm.sentinel is None:
            raise ValueError(
                f'No value found for {tm.name}, a continuous TensorMap with no sentinel value, and channel keys:{list(tm.channel_map.keys())}.',
            )
        elif missing:
            continuous_data[:] = tm.sentinel
        return fxn(continuous_data)

    return preprocess_tensor_from_file


def _weekly_alcohol(instance):
    alcohol_keys = [
        f'1568_Average-weekly-red-wine-intake_{instance}_0',
        f'1578_Average-weekly-champagne-plus-white-wine-intake_{instance}_0',
        f'1588_Average-weekly-beer-plus-cider-intake_{instance}_0',
        f'1598_Average-weekly-spirits-intake_{instance}_0',
        f'1608_Average-weekly-fortified-wine-intake_{instance}_0',
    ]

    def alcohol_from_file(tm, hd5, dependents={}):
        drinks = 0
        for k in alcohol_keys:
            data = tm.hd5_first_dataset_in_group(
                hd5, key_prefix=f'{tm.path_prefix}/{k}',
            )
            drinks += float(data[0])
        return np.array([drinks], dtype=np.float32)

    return alcohol_from_file


log_25781_2 = TensorMap(
    '25781_Total-volume-of-white-matter-hyperintensities-from-T1-and-T2FLAIR-images_2_0',
    loss='log_cosh',
    path_prefix='continuous',
    normalization={
        'mean': 7,
        'std': 8,
    },
    tensor_from_file=preprocess_with_function(np.log),
    channel_map={'white-matter-hyper-intensities': 0},
)

weight_lbs_2 = TensorMap(
    'weight_lbs',
    Interpretation.CONTINUOUS,
    normalization={
        'mean': 168.74,
        'std': 34.1,
    },
    loss='log_cosh',
    channel_map={'weight_lbs': 0},
    tensor_from_file=preprocess_with_function(
        lambda x: x * 2.20462,
        'continuous/21002_Weight_2_0',
    ),
)

weekly_alcohol_0 = TensorMap(
    'weekly_alcohol_0',
    loss='log_cosh',
    path_prefix='continuous',
    channel_map={'weekly_alcohol_0': 0},
    tensor_from_file=_weekly_alcohol(0),
)
weekly_alcohol_1 = TensorMap(
    'weekly_alcohol_1',
    loss='log_cosh',
    path_prefix='continuous',
    channel_map={'weekly_alcohol_1': 0},
    tensor_from_file=_weekly_alcohol(1),
)
weekly_alcohol_2 = TensorMap(
    'weekly_alcohol_2',
    loss='log_cosh',
    path_prefix='continuous',
    channel_map={'weekly_alcohol_2': 0},
    tensor_from_file=_weekly_alcohol(2),
)

###
weight_kg = TensorMap('weight_kg',  Interpretation.CONTINUOUS, normalization={'mean': 76.54286701805927, 'std': 15.467605416933122}, loss='log_cosh', channel_map={'weight_kg': 0})
height_cm = TensorMap('height_cm',  Interpretation.CONTINUOUS, normalization={'mean': 169.18064748408653, 'std': 9.265265197273026}, loss='log_cosh', channel_map={'height_cm': 0})
bmi_bsa = TensorMap('bmi',  Interpretation.CONTINUOUS, normalization={'mean': 26.65499238706321, 'std': 4.512077188749083}, loss='log_cosh', channel_map={'bmi': 0})

mothers_age = TensorMap(
    'mothers_age_0', Interpretation.CONTINUOUS, path_prefix='continuous',
    channel_map={'mother_age': 0, 'mother_alive': 2, 'mother_dead': 3, 'not-missing': 1},
    normalization={'mean': 75.555, 'std': 11.977}, annotation_units = 4,
)

fathers_age = TensorMap(
    'fathers_age_0', Interpretation.CONTINUOUS, path_prefix='continuous',
    channel_map={'father_age': 0, 'father_alive': 2, 'father_dead': 3, 'not-missing': 1},
    normalization={'mean':70.928, 'std': 12.746}, annotation_units = 4,
)

genetic_sex = TensorMap(
    'Genetic-sex_Male_0_0', Interpretation.CATEGORICAL, storage_type=StorageType.CATEGORICAL_INDEX,
    path_prefix='categorical', annotation_units=2, loss='categorical_crossentropy',
    channel_map={'Genetic-sex_Female_0_0': 0, 'Genetic-sex_Male_0_0': 1},
)
partition_i = 128
a_units = 128
genetic_sex_partition = TensorMap(
    'Genetic-sex_Male_0_0', Interpretation.CATEGORICAL, storage_type=StorageType.CATEGORICAL_INDEX,
    path_prefix='categorical', days_window=partition_i, annotation_units=a_units, loss='categorical_crossentropy',
    channel_map={'Genetic-sex_Female_0_0': 0, 'Genetic-sex_Male_0_0': 1},
)
age_2_partition = TensorMap(
    '21003_Age-when-attended-assessment-centre_2_0', Interpretation.CONTINUOUS, days_window=partition_i,
    annotation_units=a_units, path_prefix='continuous', loss='log_cosh', validator=make_range_validator(1, 120),
    normalization=Standardize(mean=63.358, std=7.555),
    channel_map={'21003_Age-when-attended-assessment-centre_2_0': 0},
)
sex = TensorMap(
    'Sex_Male_0_0', Interpretation.CATEGORICAL, storage_type=StorageType.CATEGORICAL_INDEX, path_prefix='categorical',
    channel_map={'Sex_Female_0_0': 0, 'Sex_Male_0_0': 1}, loss='categorical_crossentropy', annotation_units=2,
)

sex_mgb = TensorMap(
    'sex', Interpretation.CATEGORICAL, storage_type=StorageType.CATEGORICAL_INDEX, path_prefix='categorical',
    channel_map={'Sex_Female_0_0': 0, 'Sex_Male_0_0': 1}, loss='categorical_crossentropy', annotation_units=2,
)

is_male_mgb = TensorMap(
    'is_male', Interpretation.CATEGORICAL, storage_type=StorageType.CATEGORICAL_INDEX, path_prefix='categorical',
    channel_map={'Sex_Female_0_0': 0, 'Sex_Male_0_0': 1}, loss='categorical_crossentropy',
)

age_in_days = TensorMap(
    'age_in_days', Interpretation.CONTINUOUS,
    path_prefix='continuous', loss='log_cosh',
    #normalization=Standardize(mean=65, std=(1/365.0)),
    normalization=ZeroMeanStd1(),
    channel_map={'21003_Age-when-attended-assessment-centre_2_0': 0},
)
bmi = TensorMap(
    '23104_Body-mass-index-BMI_0_0', Interpretation.CONTINUOUS, path_prefix='continuous', loss='log_cosh',
    channel_map={'23104_Body-mass-index-BMI_0_0': 0}, validator=make_range_validator(0, 100),
    normalization={'mean': 27.432, 'std': 4.785},
)
bmi_ukb = TensorMap(
    'bmi', Interpretation.CONTINUOUS, path_prefix='continuous', channel_map={'23104_Body-mass-index-BMI_0_0': 0}, annotation_units=1,
    validator=make_range_validator(0, 300), normalization={'mean': 27.432061533712652, 'std': 4.785244772462738}, loss='log_cosh',
)
bmi_2 = TensorMap(
    '21001_Body-mass-index-BMI_2_0', Interpretation.CONTINUOUS, path_prefix='continuous',  loss='log_cosh',
    channel_map={'21001_Body-mass-index-BMI_2_0': 0}, validator=make_range_validator(0, 300),
    normalization=Standardize(mean=27.3397, std=4.7721),
)
bmi_as_target = TensorMap(
    'target_bmi', Interpretation.CONTINUOUS, path_prefix='continuous',  loss='log_cosh',
    channel_map={'21001_Body-mass-index-BMI_2_0': 0}, validator=make_range_validator(0, 300),
    normalization=Standardize(mean=27.3397, std=4.7721),
)
bmi_2_partition = TensorMap(
    '21001_Body-mass-index-BMI_2_0', Interpretation.CONTINUOUS, path_prefix='continuous',  loss='log_cosh',
    channel_map={'21001_Body-mass-index-BMI_2_0': 0}, validator=make_range_validator(0, 300),
    normalization=Standardize(mean=27.3397, std=4.7721), days_window=partition_i, annotation_units=a_units,
)
bmi_21_0 = TensorMap(
    '21001_Body-mass-index-BMI_0_0', Interpretation.CONTINUOUS, path_prefix='continuous', loss='log_cosh',
    channel_map={'21001_Body-mass-index-BMI_0_0': 0}, validator=make_range_validator(0, 300),
    normalization=Standardize(mean=27.3397, std=4.7721),
)
birth_year = TensorMap(
    '22200_Year-of-birth_0_0', Interpretation.CONTINUOUS, path_prefix='continuous', channel_map={'22200_Year-of-birth_0_0': 0}, annotation_units=1, loss='log_cosh',
    validator=make_range_validator(1901, 2025), normalization={'mean': 1952.0639129359386, 'std': 7.656326148519739},
)
birth_year_34 = TensorMap(
    '34_Year-of-birth_0_0', Interpretation.CONTINUOUS, path_prefix='continuous', channel_map={'34_Year-of-birth_0_0': 0}, annotation_units=1, loss='log_cosh',
    validator=make_range_validator(1901, 2025), normalization = {'mean': 1952.0639129359386, 'std': 7.656326148519739},
)
age_0 = TensorMap(
    '21003_Age-when-attended-assessment-centre_0_0', Interpretation.CONTINUOUS, path_prefix='continuous', loss='log_cosh', validator=make_range_validator(1, 120),
    normalization={'mean': 56.52847159208494, 'std': 8.095287610193827}, channel_map={'21003_Age-when-attended-assessment-centre_0_0': 0},
)
age_1 = TensorMap(
    '21003_Age-when-attended-assessment-centre_1_0', Interpretation.CONTINUOUS, path_prefix='continuous', loss='log_cosh', validator=make_range_validator(1, 120),
    normalization={'mean': 61.4476555588322, 'std': 7.3992113757847005}, channel_map={'21003_Age-when-attended-assessment-centre_1_0': 0},
)
age_2 = TensorMap(
    '21003_Age-when-attended-assessment-centre_2_0', Interpretation.CONTINUOUS,
    path_prefix='continuous', loss='log_cosh', validator=make_range_validator(1, 120),
    normalization=Standardize(mean=63.35798891483556, std=7.554638350423902),
    channel_map={'21003_Age-when-attended-assessment-centre_2_0': 0},
)

age_2_wide = TensorMap(
    'age_from_wide_csv', Interpretation.CONTINUOUS,
    path_prefix='continuous', loss='log_cosh', validator=make_range_validator(1, 120),
    normalization=Standardize(mean=63.35798891483556, std=7.554638350423902),
    channel_map={'21003_Age-when-attended-assessment-centre_2_0': 0},
)
age_2_patientage = TensorMap(
    'patientage', Interpretation.CONTINUOUS,
    path_prefix='continuous', loss='log_cosh', validator=make_range_validator(1, 120),
    normalization=Standardize(mean=63.35798891483556, std=7.554638350423902),
    channel_map={'21003_Age-when-attended-assessment-centre_2_0': 0},
)

age_2_wide = TensorMap(
    'age_from_wide_csv', Interpretation.CONTINUOUS,
    path_prefix='continuous', loss='log_cosh', validator=make_range_validator(1, 120),
    normalization=Standardize(mean=63.35798891483556, std=7.554638350423902),
    channel_map={'21003_Age-when-attended-assessment-centre_2_0': 0},
)
age_2_patientage = TensorMap(
    'patientage', Interpretation.CONTINUOUS,
    path_prefix='continuous', loss='log_cosh', validator=make_range_validator(1, 120),
    normalization=Standardize(mean=63.35798891483556, std=7.554638350423902),
    channel_map={'21003_Age-when-attended-assessment-centre_2_0': 0},
)

af_dummy = TensorMap(
    'af_in_read', Interpretation.CATEGORICAL, path_prefix='categorical', storage_type=StorageType.CATEGORICAL_FLAG,
    channel_map={'no_atrial_fibrillation': 0, 'atrial_fibrillation': 1},
)
af_dummy2 = TensorMap(
    'af_in_read', Interpretation.CATEGORICAL, path_prefix='categorical', storage_type=StorageType.CATEGORICAL_FLAG,
    channel_map={'no_af_in_read': 0, 'af_in_read': 1},
)
sex_dummy = TensorMap(
    'sex_from_wide', Interpretation.CATEGORICAL, storage_type=StorageType.CATEGORICAL_FLAG,
    path_prefix='categorical', annotation_units=2,
    channel_map={'Sex_Female_0_0': 0, 'Sex_Male_0_0': 1}, loss='categorical_crossentropy',
)
sex_dummy1 = TensorMap(
    'sex', Interpretation.CATEGORICAL, storage_type=StorageType.CATEGORICAL_FLAG,
    path_prefix='categorical', annotation_units=2,
    channel_map={'Sex_Female_0_0': 0, 'Sex_Male_0_0': 1}, loss='categorical_crossentropy',
)
sex_dummy2 = TensorMap(
    'is_female', Interpretation.CATEGORICAL, storage_type=StorageType.CATEGORICAL_FLAG,
     path_prefix='categorical', annotation_units=2,
    channel_map={'Sex_Female_0_0': 0, 'Sex_Male_0_0': 1}, loss='categorical_crossentropy',
)
sex_dummy3 = TensorMap(
    'sex_from_wide', Interpretation.CATEGORICAL, storage_type=StorageType.CATEGORICAL_FLAG,
    path_prefix='categorical', annotation_units=2,
    channel_map={'female': 0, 'male': 1}, loss='categorical_crossentropy',
)
brain_volume = TensorMap(
    '25010_Volume-of-brain-greywhite-matter_2_0', Interpretation.CONTINUOUS, path_prefix='continuous', normalization={'mean': 1165940.0, 'std': 111511.0},
    channel_map={'25010_Volume-of-brain-greywhite-matter_2_0': 0}, loss='log_cosh', loss_weight=0.1,
)

sodium = TensorMap(
    '30530_Sodium-in-urine_0_0', Interpretation.CONTINUOUS, path_prefix='continuous', channel_map={'30530_Sodium-in-urine_0_0': 0},
    normalization={'mean': 77.45323967267045, 'std': 44.441236848463774}, annotation_units=1, loss='log_cosh',
)
potassium = TensorMap(
    '30520_Potassium-in-urine_0_0', Interpretation.CONTINUOUS, path_prefix='continuous', channel_map={'30520_Potassium-in-urine_0_0': 0},
    normalization={'mean': 63.06182700345117, 'std': 33.84208704773539}, annotation_units=1, loss='log_cosh',
)
cholesterol_hdl = TensorMap(
    '30760_HDL-cholesterol_0_0', Interpretation.CONTINUOUS, path_prefix='continuous', channel_map={'30760_HDL-cholesterol_0_0': 0},
    normalization={'mean': 1.4480129055069355, 'std': 0.3823115953478376}, annotation_units=1, loss='log_cosh',
)
cholesterol = TensorMap(
    '30690_Cholesterol_0_0', Interpretation.CONTINUOUS, path_prefix='continuous', channel_map={'30690_Cholesterol_0_0': 0},
    normalization={'mean': 5.692381214399044, 'std': 1.1449409331668705}, annotation_units=1, loss='log_cosh',
)

cigarettes = TensorMap('2887_Number-of-cigarettes-previously-smoked-daily_0_0', Interpretation.CONTINUOUS, path_prefix='continuous', channel_map={'2887_Number-of-cigarettes-previously-smoked-daily_0_0': 0}, normalization = {'mean': 18.92662147068755, 'std':10.590930376362259}, annotation_units=1)
alcohol = TensorMap('5364_Average-weekly-intake-of-other-alcoholic-drinks_0_0', Interpretation.CONTINUOUS, path_prefix='continuous', channel_map={'5364_Average-weekly-intake-of-other-alcoholic-drinks_0_0': 0}, normalization = {'mean': 0.03852570253005904, 'std':0.512608370266108}, annotation_units=1)


def alcohol_channel_map(instance=0, array_idx=0):
    return {
        f'Alcohol-intake-frequency_Never_{instance}_{array_idx}': 0,
        f'Alcohol-intake-frequency_Special-occasions-only_{instance}_{array_idx}': 1,
        f'Alcohol-intake-frequency_One-to-three-times-a-month_{instance}_{array_idx}': 2,
        f'Alcohol-intake-frequency_Once-or-twice-a-week_{instance}_{array_idx}': 3,
        f'Alcohol-intake-frequency_Three-or-four-times-a-week_{instance}_{array_idx}': 4,
        f'Alcohol-intake-frequency_Daily-or-almost-daily_{instance}_{array_idx}': 5,
    }


alcohol_0 = TensorMap('alcohol_0', Interpretation.CATEGORICAL, path_prefix='categorical', channel_map=alcohol_channel_map(instance=0))
alcohol_1 = TensorMap('alcohol_1', Interpretation.CATEGORICAL, path_prefix='categorical', channel_map=alcohol_channel_map(instance=1))
alcohol_2 = TensorMap('alcohol_2', Interpretation.CATEGORICAL, path_prefix='categorical', channel_map=alcohol_channel_map(instance=2))


def alcohol_status_map(instance=0, array_idx=0):
    return {
        f'Alcohol-drinker-status_Never_{instance}_{array_idx}': 0,
        f'Alcohol-drinker-status_Previous_{instance}_{array_idx}': 1,
        f'Alcohol-drinker-status_Current_{instance}_{array_idx}': 2,
    }


alcohol_status_0 = TensorMap('alcohol_status_0', Interpretation.CATEGORICAL, path_prefix='categorical', channel_map=alcohol_status_map(instance=0))
alcohol_status_1 = TensorMap('alcohol_status_1', Interpretation.CATEGORICAL, path_prefix='categorical', channel_map=alcohol_status_map(instance=1))
alcohol_status_2 = TensorMap('alcohol_status_2', Interpretation.CATEGORICAL, path_prefix='categorical', channel_map=alcohol_status_map(instance=2))


def alcohol_meals_map(instance=0, array_idx=0):
    return {
        f'Alcohol-usually-taken-with-meals_No_{instance}_{array_idx}': 0,
        f'Alcohol-usually-taken-with-meals_It-varies_{instance}_{array_idx}': 1,
        f'Alcohol-usually-taken-with-meals_Yes_{instance}_{array_idx}': 2,
    }


alcohol_meals_0 = TensorMap('alcohol_meals_0', Interpretation.CATEGORICAL, path_prefix='categorical', channel_map=alcohol_meals_map(instance=0))
alcohol_meals_1 = TensorMap('alcohol_meals_1', Interpretation.CATEGORICAL, path_prefix='categorical', channel_map=alcohol_meals_map(instance=1))
alcohol_meals_2 = TensorMap('alcohol_meals_2', Interpretation.CATEGORICAL, path_prefix='categorical', channel_map=alcohol_meals_map(instance=2))

coffee = TensorMap(
    '1498_Coffee-intake_0_0', Interpretation.CONTINUOUS, path_prefix='continuous', channel_map={'1498_Coffee-intake_0_0': 0},
    normalization={'mean': 2.015086529948216, 'std': 2.0914960998390497}, annotation_units=1,
)
water = TensorMap(
    '1528_Water-intake_0_0', Interpretation.CONTINUOUS, path_prefix='continuous', channel_map={'1528_Water-intake_0_0': 0},
    normalization={'mean': 2.7322977785723324, 'std': 2.261996814128837}, annotation_units=1,
)
meat = TensorMap(
    '3680_Age-when-last-ate-meat_0_0', Interpretation.CONTINUOUS, path_prefix='continuous',
    channel_map={'3680_Age-when-last-ate-meat_0_0': 0},
    normalization={'mean': 29.74062983480561, 'std': 14.417292213873964}, annotation_units=1,
)
walks = TensorMap(
    '864_Number-of-daysweek-walked-10-minutes_0_0', Interpretation.CONTINUOUS, path_prefix='continuous',
    channel_map={'864_Number-of-daysweek-walked-10-minutes_0_0': 0},
    normalization={'mean': 5.369732285440756, 'std': 1.9564911925721618}, annotation_units=1,
)
walk_duration = TensorMap(
    '874_Duration-of-walks_0_0', Interpretation.CONTINUOUS, path_prefix='continuous', channel_map={'874_Duration-of-walks_0_0': 0},
    normalization={'mean': 61.64092215093373, 'std': 78.79522990818906}, annotation_units=1,
)
physical_activities = TensorMap(
    '884_Number-of-daysweek-of-moderate-physical-activity-10-minutes_0_0', Interpretation.CONTINUOUS, path_prefix='continuous',
    channel_map={'884_Number-of-daysweek-of-moderate-physical-activity-10-minutes_0_0': 0},
    normalization={'mean': 3.6258833281089258, 'std': 2.3343738999823676}, annotation_units=1,
)
physical_activity = TensorMap(
    '894_Duration-of-moderate-activity_0_0', Interpretation.CONTINUOUS, path_prefix='continuous',
    channel_map={'894_Duration-of-moderate-activity_0_0': 0},
    normalization={'mean': 66.2862593866103, 'std': 77.28681218835422}, annotation_units=1,
)
physical_activity_vigorous = TensorMap(
    '904_Number-of-daysweek-of-vigorous-physical-activity-10-minutes_0_0', Interpretation.CONTINUOUS,
    channel_map={'904_Number-of-daysweek-of-vigorous-physical-activity-10-minutes_0_0': 0}, path_prefix='continuous',
    normalization={'mean': 1.838718301735063, 'std': 1.9593505421480895}, annotation_units=1,
)
physical_activity_vigorous_duration = TensorMap(
    '914_Duration-of-vigorous-activity_0_0', Interpretation.CONTINUOUS, path_prefix='continuous',
    channel_map={'914_Duration-of-vigorous-activity_0_0': 0},
    normalization={'mean': 44.854488382965144, 'std': 48.159967071781466}, annotation_units=1,
)
tv = TensorMap(
    '1070_Time-spent-watching-television-TV_0_0', Interpretation.CONTINUOUS, path_prefix='continuous',
    channel_map={'1070_Time-spent-watching-television-TV_0_0': 0},
    normalization={'mean': 2.7753595642790914, 'std': 1.7135478462887321}, annotation_units=1,
)
computer = TensorMap(
    '1080_Time-spent-using-computer_0_0', Interpretation.CONTINUOUS, path_prefix='continuous',
    channel_map={'1080_Time-spent-using-computer_0_0': 0},
    normalization={'mean': 0.9781465855433753, 'std': 1.4444414103121512}, annotation_units=1,
)
car = TensorMap(
    '1090_Time-spent-driving_0_0', Interpretation.CONTINUOUS, path_prefix='continuous', channel_map={'1090_Time-spent-driving_0_0': 0},
    normalization={'mean': 0.8219851505445748, 'std': 1.304094814200189}, annotation_units=1,
)
summer = TensorMap(
    '1050_Time-spend-outdoors-in-summer_0_0', Interpretation.CONTINUOUS, path_prefix='continuous',
    channel_map={'1050_Time-spend-outdoors-in-summer_0_0': 0},
    normalization={'mean': 3.774492304870845, 'std': 2.430483731404539}, annotation_units=1,
)
winter = TensorMap(
    '1060_Time-spent-outdoors-in-winter_0_0', Interpretation.CONTINUOUS, path_prefix='continuous',
    channel_map={'1060_Time-spent-outdoors-in-winter_0_0': 0},
    normalization={'mean': 1.8629686916635555, 'std': 1.88916218603397}, annotation_units=1,
)

systolic_blood_pressure_0 = TensorMap(
    '4080_Systolic-blood-pressure-automated-reading_0_0', Interpretation.CONTINUOUS, path_prefix='continuous', loss='log_cosh',
    channel_map={'4080_Systolic-blood-pressure-automated-reading_0_0': 0}, validator=make_range_validator(40, 400),
    normalization={'mean': 137.79964191990328, 'std': 19.292863700283757},
)
diastolic_blood_pressure_0 = TensorMap(
    '4079_Diastolic-blood-pressure-automated-reading_0_0', Interpretation.CONTINUOUS, path_prefix='continuous', loss='log_cosh',
    channel_map={'4079_Diastolic-blood-pressure-automated-reading_0_0': 0}, validator=make_range_validator(20, 300),
    normalization={'mean': 82.20657551284782, 'std': 10.496040770224475},
)

systolic_blood_pressure_1 = TensorMap(
    '4080_Systolic-blood-pressure-automated-reading_1_0', Interpretation.CONTINUOUS, path_prefix='continuous', loss='log_cosh',
    channel_map={'4080_Systolic-blood-pressure-automated-reading_1_0': 0}, validator=make_range_validator(40, 400),
    normalization={'mean': 137.79964191990328, 'std': 19.292863700283757},
)
diastolic_blood_pressure_1 = TensorMap(
    '4079_Diastolic-blood-pressure-automated-reading_1_0', Interpretation.CONTINUOUS, path_prefix='continuous', loss='log_cosh',
    channel_map={'4079_Diastolic-blood-pressure-automated-reading_1_0': 0}, validator=make_range_validator(20, 300),
    normalization={'mean': 82.20657551284782, 'std': 10.496040770224475},
)

systolic_blood_pressure_2 = TensorMap(
    '4080_Systolic-blood-pressure-automated-reading_2_0', Interpretation.CONTINUOUS, path_prefix='continuous', loss='log_cosh',
    channel_map={'4080_Systolic-blood-pressure-automated-reading_2_0': 0}, validator=make_range_validator(40, 400),
    normalization={'mean': 137.79964191990328, 'std': 19.292863700283757},
)
diastolic_blood_pressure_2 = TensorMap(
    '4079_Diastolic-blood-pressure-automated-reading_2_0', Interpretation.CONTINUOUS, path_prefix='continuous', loss='log_cosh',
    channel_map={'4079_Diastolic-blood-pressure-automated-reading_2_0': 0}, validator=make_range_validator(20, 300),
    normalization={'mean': 82.20657551284782, 'std': 10.496040770224475},
)

hypertension = TensorMap(
    'hypertension', Interpretation.CATEGORICAL,
    storage_type=StorageType.CATEGORICAL_INDEX, path_prefix='categorical',
    loss='categorical_crossentropy',
    channel_map={'no_hypertension': 0, 'hypertension': 1},
)

hypertension_diagnosis = TensorMap(
    'hypertension_diagnosis', Interpretation.CATEGORICAL,
    storage_type=StorageType.CATEGORICAL_INDEX, path_prefix='categorical',
    loss='categorical_crossentropy',
    channel_map={'no_hypertension': 0, 'hypertension': 1},
)

hypertension_icd_bp = TensorMap(
    'hypertension_icd_bp', Interpretation.CATEGORICAL,
    storage_type=StorageType.CATEGORICAL_INDEX, path_prefix='categorical',
    loss='categorical_crossentropy',
    channel_map={'no_hypertension': 0, 'hypertension': 1},
)
htn_icd_bp = TensorMap(
    'htn_icd_bp', Interpretation.CATEGORICAL,
    storage_type=StorageType.CATEGORICAL_INDEX, path_prefix='categorical',
    loss='categorical_crossentropy',
    channel_map={'no_hypertension': 0, 'hypertension': 1},
)

diabetes = TensorMap(
    'dm', Interpretation.CATEGORICAL,
    storage_type=StorageType.CATEGORICAL_INDEX, path_prefix='categorical',
    loss='categorical_crossentropy',
    channel_map={'no_diabetes_type_2': 0, 'diabetes_type_2': 1},
)

hypercholesterolemia = TensorMap(
    'hypercholesterolemia', Interpretation.CATEGORICAL,
    storage_type=StorageType.CATEGORICAL_INDEX, path_prefix='categorical',
    loss='categorical_crossentropy',
    channel_map={'no_hypercholesterolemia': 0, 'hypercholesterolemia': 1},
)

hypertension_med = TensorMap(
    'start_fu_hypertension_med', Interpretation.CATEGORICAL, loss='categorical_crossentropy',
    storage_type=StorageType.CATEGORICAL_INDEX, path_prefix='categorical',
    channel_map={'no_start_fu_hypertension_med':0, 'start_fu_hypertension_med': 1},
)
peak_vo2 = TensorMap(
    'peak_vo2', Interpretation.CONTINUOUS,
    storage_type=StorageType.CATEGORICAL_INDEX, path_prefix='continuous',
    channel_map={'peak_vo2':0},
)
cad = TensorMap(
    'cad', Interpretation.CATEGORICAL, loss='categorical_crossentropy',
    storage_type=StorageType.CATEGORICAL_INDEX, path_prefix='categorical',
    channel_map={'no_coronary_artery_disease':0, 'coronary_artery_disease': 1},
)

target_bmi = TensorMap(
    'target_bmi', Interpretation.CONTINUOUS, shape=(1,),
)
