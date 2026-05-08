import datetime
from numbers import Number
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import tensorflow as tf

from ml4ht.data.data_description import DataDescription

from data_descriptions.echo import VIEW_OPTION_KEY


class EcholabDataDescription(DataDescription):
    # DataDescription for a wide file

    def __init__(
            self,
            wide_df: pd.DataFrame,
            sample_id_column: str,
            column_names: str,
            name: str,
            categories: Dict = None,
            cls_categories_map: Dict = None,
            survival_task_configs: Optional[List[Dict]] = None,
            transforms=None,
    ):
        """
        """
        self.wide_df = wide_df
        self._name = name
        self.sample_id_column = sample_id_column
        self.column_names = column_names
        self.categories = categories
        self.prep_df()
        self.transforms = transforms or []
        self.cls_categories_map = cls_categories_map
        self.survival_task_configs = survival_task_configs or []

    @staticmethod
    def _to_timedelta_age(age_value):
        if isinstance(age_value, pd.Timedelta):
            return age_value
        if isinstance(age_value, np.timedelta64):
            return pd.to_timedelta(age_value)
        if isinstance(age_value, Number):
            return pd.to_timedelta(float(age_value), unit='D')
        return pd.to_timedelta(age_value)

    def _survival_tensor_from_row(self, row: pd.Series, config: Dict) -> np.ndarray:
        assess_date = pd.to_datetime(row[config['date_column']]).to_pydatetime()
        has_disease = int(row[config['event_column']])
        assess_age = self._to_timedelta_age(row[config['age_column']])
        censor_age = self._to_timedelta_age(row[config['censor_age_column']])
        event_age = self._to_timedelta_age(row[config['event_age_column']]) if has_disease else censor_age

        if has_disease and event_age <= assess_age:
            if config['incidence_only']:
                raise ValueError(
                    f"{config['event_column']} only considers incident diagnoses for sample_id {row[self.sample_id_column]}."
                )
            censor_age = event_age

        follow_up = (event_age - assess_age) if has_disease else (censor_age - assess_age)
        censor_date = assess_date + follow_up
        days_per_interval = 365.25 * config['follow_up_years'] / config['intervals']
        survival_then_censor = np.zeros(config['intervals'] * 2, dtype=np.float32)

        for i, day_delta in enumerate(np.arange(0, 365.25 * config['follow_up_years'], days_per_interval)):
            cur_date = assess_date + datetime.timedelta(days=day_delta)
            survival_then_censor[i] = float(cur_date < censor_date)
            survival_then_censor[config['intervals'] + i] = has_disease * float(
                censor_date <= cur_date < censor_date + datetime.timedelta(days=days_per_interval),
            )

        if has_disease and event_age <= assess_age and not config['incidence_only']:
            survival_then_censor[config['intervals']] = has_disease
        return survival_then_censor

    def prep_df(self):
        self.wide_df.index = self.wide_df[self.sample_id_column]
        self.wide_df = self.wide_df.drop_duplicates()

    def get_loading_options(self, sample_id):
        row = self.wide_df.loc[sample_id]

        # a loading option is a dictionary of options to use at loading time
        # we use DATE_OPTION_KEY to make the date selection utilities work
        loading_options = [{VIEW_OPTION_KEY: row}]

        # it's get_loading_options, not get loading_option, so we return a list
        return loading_options

    def get_raw_data(self, sample_id, loading_option=None):
        try:
            if sample_id.shape[0] > 1:
                sample_id = sample_id[0]
        except AttributeError:
            pass
        try:
            sample_id = sample_id.decode('UTF-8')
        except (UnicodeDecodeError, AttributeError):
            pass
        row = self.wide_df.loc[sample_id]
        data = row[self.column_names].values
        label_noise = np.zeros(len(self.column_names))
        for transform in self.transforms:
            label_noise += transform()
        if self.categories:
            output_data = np.zeros(len(self.categories), dtype=np.float32)
            output_data[self.categories[data[0]]['index']] = 1.0
            return output_data
        # ---------- Adaptation for regression + classification ---------- #
        if self.cls_categories_map or self.survival_task_configs:
            # If training include classification tasks:
            data = []
            excluded_columns = []
            if self.cls_categories_map:
                excluded_columns.extend(self.cls_categories_map['cls_output_order'])
            excluded_columns.extend([cfg['event_column'] for cfg in self.survival_task_configs])
            reg_data = row[self.column_names].drop(excluded_columns, errors='ignore').values
            if len(reg_data) > 0:
                data.append(np.array(reg_data, dtype=np.float32))

            if self.cls_categories_map:
                for k in self.cls_categories_map['cls_output_order']:
                    # Changing values to class labels:
                    row_cls_lbl = self.cls_categories_map[k][row[k]]
                    # Changing class indices to one hot vectors
                    cls_one_hot = tf.keras.utils.to_categorical(
                        row_cls_lbl,
                        num_classes=len(self.cls_categories_map[k]),
                    )
                    data.append(cls_one_hot)

            for config in self.survival_task_configs:
                data.append(self._survival_tensor_from_row(row, config))

            if len(data) == 1:
                data = data[0]

            return data
        # ---------------------------------------------------------------- #
        return np.array(data, dtype=np.float32)

    @property
    def name(self):
        # if we have multiple wide file DataDescriptions at the same time,
        # this will allow us to differentiate between them
        return self._name
