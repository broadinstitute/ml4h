from ml4ht.data.defines import SampleID

import os
import glob
from typing import Callable, List, Union, Optional, Tuple, Dict, Any

import h5py
import numcodecs
import numpy as np
import pandas as pd

from ml4ht.data.data_description import DataDescription
from ml4ht.data.util.date_selector import DATE_OPTION_KEY
from ml4ht.data.defines import LoadingOption, Tensor

from ml4h.TensorMap import TensorMap
from ml4h.defines import PARTNERS_DATETIME_FORMAT, ECG_REST_AMP_LEADS


def sample_id_from_path(hd5_path: str) -> int:
    return int(os.path.splitext(os.path.basename(hd5_path))[0])


# this is used to load data from our hd5 storage
def decompress_data(data_compressed: np.array, dtype: str) -> np.array:
    """Decompresses a compressed byte array. If the primitive type of the data
    to decompress is a string, calls decode using the zstd codec. If the
    primitive type of the data to decompress is not a string (e.g. int or
    float), the buffer is interpreted using the passed dtype."""
    codec = numcodecs.zstd.Zstd()
    data_decompressed = codec.decode(data_compressed)
    if dtype == "str":
        data = data_decompressed.decode()
    else:
        data = np.frombuffer(data_decompressed, dtype)
    return data


class ECGDataDescription(DataDescription):
    S3_PATH_OPTION = "s3_path"
    TEXT_DIAGNOSES = {
        "sb": ["sinus brady"],
        "st": ["sinus tachy"],
        "af": [
            "afib",
            "atrial fib",
            "afibrillation",
            "atrialfibrillation",
        ],
        "rbbb": [
            "right bbb",
            "rbbb",
            "right bundle branch block",
        ],
        "lbbb": [
            "left bbb",
            "lbbb",
            "left bundle branch block",
        ],
        "avb": [
            "1st degree atrioventricular block",
            "1st degree av block",
            "first degree av block",
            "first degree atrioventricular block",
        ],
        "lvh": [
            "biventricular hypertrophy",
            "biventriclar hypertrophy",
            "leftventricular hypertrophy",
            "combined ventricular hypertrophy",
            "left ventricular hypertr",
        ],
        "ischemia": [
            "diffuse st segment elevation",
            "consistent with lateral ischemia",
            "subendocardial ischemia",
            "apical subendocardial ischemia",
            "inferior subendocardial ischemia",
            "anterolateral ischemia",
            "antero-apical ischemia",
            "consider anterior and lateral ischemia",
            "st segment depression",
            "minor st segment depression",
            "st segment depression in leads v4-v6",
            "anterolateral st segment depression",
            "infero- st segment depression",
            "st depression",
            "suggest anterior ischemia",
            "st segment depression is more marked in leads",
            "possible anterior wall ischemia",
            "consistent with ischemia",
            "diffuse scooped st segment depression",
            "anterolateral subendocardial ischemia",
            "diffuse st segment depression",
            "st segment elevation consistent with acute injury",
            "inferior st segment elevation and q waves",
            "st segment depression in anterolateral leads",
            "widespread st segment depression",
            "consider anterior ischemia",
            "suggesting anterior ischemia",
            "consistent with subendocardial ischemia",
            "marked st segment depression in leads",
            "inferior st segment depression",
            "st segment elevation in leads",
            "st segment elevation",
            "st segment depressions more marked",
            "anterior st segment depression",
            "apical st depression",
            "septal ischemia",
            "st segment depression in leads",
            "suggests anterolateral ischemia",
            "st elevation",
            "diffuse elevation of st segments",
            "marked st segment depression",
            "anterior infarct or transmural ischemia",
            "inferoapical st segment depression",
            "lateral ischemia",
            "nonspecific st segment depression",
            "anterior subendocardial ischemia",
        ],
    }
    META_DATA_FIELDS = [
        "gender",
        "patientage",
        "locationname",
        "waveform_samplebase",
        "ventricularrate_md",
        "qrsduration_md",
        "printerval_md",
        "qtinterval_md",
        "paxis_md",
        "raxis_md",
        "taxis_md",
        "weightlbs",
        "heightin",
    ]

    def __init__(
            self,
            local_hd5_folder: str,
            name: str,
            ecg_len: int,
            transforms: List[Callable[[Tensor, LoadingOption], Tensor]] = None,
            date_format: str = PARTNERS_DATETIME_FORMAT,
            hd5_path_to_ecg: str = "partners_ecg_rest",
            leads: Dict[str, int] = ECG_REST_AMP_LEADS,
    ):
        """
        Reads ECGs in our hd5 format stored on ccds s3.
        If s3 information is provided, if an hd5 is not available locally,
        it's downloaded from s3.
        :param local_hd5_folder: Where to check for locally stored hd5s.
            Also where hd5s will be downloaded to from s3.
        :param name: Name of the output of this DataDescription.
        :param ecg_len: length in samples to interpolate all leads of ECG to.
        :param transforms: transformations including augmentations to apply to ECG.
        :param hd5_path_to_ecg: key in hd5 of the ECG leads
        :param leads: mapping from lead name in hd5 -> channel in output array
        """
        self.local_hd5_folder = local_hd5_folder
        self.date_format = date_format
        self.hd5_path_to_ecg = hd5_path_to_ecg
        self._name = name
        self.transforms = transforms or []
        self.ecg_len = ecg_len
        self.leads = leads
        # s3
        self.local_only = True

    def _prep_s3_bucket_paths(
            self,
            s3_bucket_paths: Optional[Union[str, List[str]]],
    ) -> List[str]:
        """Prepares user input s3 paths for internal use"""
        if s3_bucket_paths is None:
            return
        elif isinstance(s3_bucket_paths, list):
            paths = s3_bucket_paths
        elif isinstance(s3_bucket_paths, str):
            paths = [s3_bucket_paths]
        else:
            raise TypeError(
                f"Cannot use s3_bucket_path input of type {type(s3_bucket_paths)}.",
            )
        for path in paths:
            os.makedirs(self._local_path(path), exist_ok=True)
        return paths

    def _local_path(self) -> str:
        """The folder downloaded hd5s from s3_bucket_path end up in"""
        if self.local_only:
            return self.local_hd5_folder

    def local_sample_ids(self) -> int:
        yield from map(
            sample_id_from_path,
            glob.glob(os.path.join(self.local_hd5_folder, "*.hd5")),
        )

    def _loading_options(
            self,
            sample_id: int,
            local_hd5_folder: str,
    ) -> List[LoadingOption]:
        hd5_path = os.path.join(local_hd5_folder, f"{sample_id}.hd5")
        with h5py.File(hd5_path, "r") as hd5:
            dates = list(
                hd5[self.hd5_path_to_ecg],
            )  # list all of the dates of saved ECGs
            sites = [
                decompress_data(
                    data_compressed=hd5[f'{self.hd5_path_to_ecg}/{date}/sitename'][()],
                    dtype='str'
                )
                for date in dates
            ]

        # print(f'got sites {sites} and dates {dates}')

        return [
            {
                'SITE': site,
                DATE_OPTION_KEY: pd.to_datetime(date).to_pydatetime(),
            }
            for date, site in zip(dates, sites)
        ]

    def get_loading_options(self, sample_id) -> List[LoadingOption]:
        options = []
        if self.local_only:
            return self._loading_options(sample_id, self.local_hd5_folder)
        return options

    def get_raw_data(self, sample_id, loading_option: LoadingOption) -> Tensor:
        hd5_path = os.path.join(self._local_path(), f"{sample_id}.hd5")
        with h5py.File(hd5_path, "r") as hd5:
            # we have to use the date provided in the loading option to get data from the hd5
            date_str = loading_option[DATE_OPTION_KEY].strftime(
                self.date_format,
            )

            # use the date to navigate to leads of the ECG
            compressed_leads = {
                lead: hd5[self.hd5_path_to_ecg][date_str][lead] for lead in self.leads
            }

            # decompress the ECG once retrieved
            ecg = np.zeros((self.ecg_len, len(self.leads)), dtype=np.float32)
            for lead in compressed_leads:
                voltages = decompress_data(
                    compressed_leads[lead][()],
                    dtype=compressed_leads[lead].attrs["dtype"],
                )
                if voltages.shape[0] != self.ecg_len:
                    voltages = np.interp(
                        np.linspace(0, 1, self.ecg_len),
                        np.linspace(0, 1, voltages.shape[0]),
                        voltages,
                    )
                ecg[:, self.leads[lead]] = voltages
            for transform in self.transforms:
                ecg = transform(ecg, loading_option)
            return ecg

    @property
    def name(self):
        return self._name

    def process_read(self, read: str) -> Dict[str, bool]:
        # TODO: speed this up, does a for loop over read for every kw
        return {
            name: any(kw in read for kw in kws)
            for name, kws in self.TEXT_DIAGNOSES.items()
        }

    def get_meta_data(self, hd5: h5py.Dataset) -> Dict[str, Any]:
        """hd5 should be keyed to desired ECG date"""
        out = {}
        for key in self.META_DATA_FIELDS:
            val = np.nan
            if key in hd5:
                dset = hd5[key]
                val = decompress_data(dset[()], dset.attrs["dtype"])
            out[key] = val
        return out

    def get_summary_data(self, sample_id, loading_option):
        """
        This function lets us pair down the 5000 x 12 data into something manageable
        for a DataFrame built by data exploration functions
        """
        # ECG meta data
        hd5_path = os.path.join(self._local_path(), f"{sample_id}.hd5")
        with h5py.File(hd5_path, "r") as hd5:
            date_str = loading_option[DATE_OPTION_KEY].strftime(
                self.date_format,
            )
            # get meta data
            out = self.get_meta_data(hd5[self.hd5_path_to_ecg][date_str])
            if "read_md_clean" in hd5[self.hd5_path_to_ecg][date_str]:
                read = hd5[self.hd5_path_to_ecg][date_str]["read_md_clean"]
                read = decompress_data(read[()], read.attrs["dtype"]).lower()
                out["read"] = "md"

            elif "read_pc_clean" in hd5[self.hd5_path_to_ecg][date_str]:
                read = hd5[self.hd5_path_to_ecg][date_str]["read_pc_clean"]
                read = decompress_data(read[()], read.attrs["dtype"]).lower()
                out["read"] = "pc"
            else:
                read = ""
                out["read"] = "none"

        # Raw ECG information
        ecg = self.get_raw_data(sample_id, loading_option)
        out["max_absolute_amp"] = np.abs(ecg).max()
        out["num_zeros"] = np.count_nonzero(ecg == 0)

        for lead_name, lead_idx in ECG_REST_AMP_LEADS.items():
            out[f"{lead_name}_zeros"] = np.count_nonzero(ecg[..., lead_idx] == 0)
            out = {**out, **self.process_read(read)}
        return out


class DataFrameDataDescription(DataDescription):
    def __init__(
            self,
            df: pd.DataFrame,
            col: str,
            process_col: Callable[[Any], Tensor] = None,
            name: str = None,
    ):
        """
        Gets data from a column of the provided DataFrame.
        :param col: The column name to get data from
        :param process_col: Function to turn the column value into Tensor
        :param name: Optional overwrite of the df column name
        """
        self.process_col = process_col or self._default_process_call
        self.df = df
        self.col = col
        self._name = name or col

    @staticmethod
    def _default_process_call(x: Any) -> Tensor:
        return np.array(x)

    @property
    def name(self) -> str:
        return self._name

    def get_loading_options(self, sample_id):
        row = self.df.loc[sample_id]
        return [{
            'start_fu_datetime': pd.to_datetime(row['start_fu_datetime']),
        }]

    def get_raw_data(
            self,
            sample_id: SampleID,
            loading_option: LoadingOption,
    ) -> Tensor:
        col_val = self.df.loc[sample_id][self.col]
        if self.col == 'age_in_days' and 'day_delta' in loading_option:
            col_val -= loading_option['day_delta']
        return self.process_col(col_val)


def one_hot_sex(x):
    return np.array([1, 0], dtype=np.float32) if x in [0, "Female"] else np.array([0, 1], dtype=np.float32)


def make_zscore(mu, std):
    def zscore(x):
        return (x-mu) / (1e-8+std)
    return zscore


def dataframe_data_description_from_tensor_map(
        tensor_map: TensorMap,
        dataframe: pd.DataFrame,
        is_input: bool = False,
) -> DataDescription:
    if tensor_map.is_categorical():
        process_col = one_hot_sex
    else:
        process_col = make_zscore(dataframe[tensor_map.name].mean(), dataframe[tensor_map.name].std())
    return DataFrameDataDescription(
        dataframe,
        col = tensor_map.name,
        process_col = process_col,
        name = tensor_map.input_name() if is_input else tensor_map.output_name(),
    )


class SurvivalWideFile(DataDescription):
    # DataDescription for a wide file

    def __init__(
            self,
            name: str,
            wide_df: pd.DataFrame,
            intervals: int = 25,
            follow_up_years: int = 10,
            event_age: str = 'hf_nlp_age',
            event_column: str = 'hf_nlp_event',
            ecg_age_column: str = 'ecg_age',
            start_age_column: str = 'start_fu_age',
            end_age_column: str = 'last_encounter',
    ):
        """
        """
        self.name = name
        self.intervals = intervals

        self.wide_df = wide_df
        self.event_age = event_age
        self.event_column = event_column
        self.ecg_age_column = ecg_age_column
        self.end_age_column = end_age_column
        self.follow_up_years = follow_up_years
        self.start_age_column = start_age_column
        print(f'Survival Curve with {365.25 * self.follow_up_years / self.intervals:.1f} days per interval.')

    def get_loading_options(self, sample_id):
        row = self.wide_df.loc[sample_id]
        ecg_date = pd.to_datetime(row[DATE_OPTION_KEY])
        start_date = ecg_date + (
                    pd.to_timedelta(row[self.start_age_column]) - pd.to_timedelta(row[self.ecg_age_column]))
        return [{
            DATE_OPTION_KEY: ecg_date,
            'start_date': start_date,
            'start_age': row[self.start_age_column],
            'event_age': row[self.event_age],
            'event': row[self.event_column],
        }]

    def get_raw_data(self, sample_id, loading_option):
        """expects time of ECG in the loading option as DATE_OPTION_KEY"""
        ecg_date = loading_option[DATE_OPTION_KEY]
        start_date = loading_option['start_date']
        ecg_age_difference = ecg_date - start_date

        row = self.wide_df.loc[sample_id]
        ecg_age = row[self.start_age_column] + ecg_age_difference

        has_disease = row[self.event_column]

        #         print(f"""
        #               OKAY: start_date {start_date} ecg_date {ecg_date} ecg_age_difference {ecg_age_difference}
        #               start age: {row[self.start_age_column]} ecg_age {ecg_age} has_disease: {has_disease}
        #               event age: {row[self.event_age]}, last encounter age: {row[self.end_age_column]}
        #               """)
        if has_disease:
            follow_up = pd.to_timedelta(row[self.event_age]) - ecg_age
        else:
            follow_up = pd.to_timedelta(row[self.end_age_column]) - ecg_age

        censor_date = ecg_date + follow_up
        days_per_interval = 365.25 * self.follow_up_years / self.intervals
        survival_then_censor = np.zeros(self.intervals * 2, dtype=np.float32)

        for i, day_delta in enumerate(np.arange(0, 365.25 * self.follow_up_years, days_per_interval)):
            cur_date = ecg_date + datetime.timedelta(days=day_delta)
            survival_then_censor[i] = float(cur_date < censor_date)
            survival_then_censor[self.intervals + i] = has_disease * float(
                censor_date <= cur_date < censor_date + datetime.timedelta(days=days_per_interval))
        # Handle prevalent diseases
        if has_disease and pd.to_timedelta(row[self.event_age]) <= pd.to_timedelta(ecg_age):
            survival_then_censor[self.intervals] = has_disease
        return survival_then_censor

    def name(self):
        return self.name
