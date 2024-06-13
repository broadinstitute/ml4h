from typing import Dict

import numpy as np
import pandas as pd
import tensorflow as tf

from ml4ht.data.data_description import DataDescription

from data_descriptions.echo import VIEW_OPTION_KEY
from model_descriptions.echo import ONE_HOT_2CLS


class EcholabDataDescription(DataDescription):
    # DataDescription for a wide file

    def __init__(
            self,
            wide_df: pd.DataFrame,
            sample_id_column: str,
            column_names: list,
            name: str,
            categories: Dict = None,
            cls_categories_map: Dict = None,
            transforms=None,
            survival_names=[],
            survival_intervals=25,
            survival_days_window=3650
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
        self.survival_names = survival_names
        self.survival_intervals = survival_intervals
        self.survival_days_window = survival_days_window

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
        if (not self.survival_names) and (not self.cls_categories_map):
            return np.squeeze(np.array(row[self.column_names].values, dtype=np.float32))
        # label_noise = np.zeros(len(self.column_names))
        # for transform in self.transforms:
        #     label_noise += transform()
        # if self.categories:
        #     output_data = np.zeros(len(self.categories), dtype=np.float32)
        #     output_data[self.categories[data[0]]['index']] = 1.0
        #     return output_data
        # ---------- Adaptation for regression + classification + survival ---------- #
        data = []
        if self.survival_names:
            # If training include survival curve tasks:
            for s_name in self.survival_names:
                data.append(self.make_survival_curve(s_name, row))
                # print(data[0].shape)
        survival_columns = [x for s_name in self.survival_names for x in [s_name + '_event', s_name + '_follow_up']]
        if set(self.column_names) != set(self.cls_categories_map['cls_output_order']+survival_columns):
            # If training include regression tasks:
            reg_data = row[self.column_names].drop(self.cls_categories_map['cls_output_order']+survival_columns).values
            data.append(np.squeeze(np.array(reg_data, dtype=np.float32)))
        if self.cls_categories_map:
            # If training include classification tasks:
            for k in self.cls_categories_map['cls_output_order']:
                # Changing values to class labels:
                # if row[k].size > 1:
                #     print(row[k])
                #     row_cls_lbl = [self.cls_categories_map[k][row[k][i]] for i in range(row[k].size)]
                # else:
                row_cls_lbl = self.cls_categories_map[k][row[k]]
                if ONE_HOT_2CLS or len(self.cls_categories_map[k]) > 2:
                    # Changing class indices to one hot vectors
                    cls_one_hot = tf.keras.utils.to_categorical(row_cls_lbl,
                                                                num_classes=len(self.cls_categories_map[k]))
                else:
                    cls_one_hot = row_cls_lbl
                data.append(cls_one_hot)

        if len(data) == 1:
            data = data[0]

        return data
        # ---------------------------------------------------------------- #


    def make_survival_curve(self, hf_name, row):
        intervals = self.survival_intervals
        has_disease = row[hf_name+'_event']
        follow_up_days = row[hf_name+'_follow_up'] - 30  # Adaptation for a 30 day "blanking" period (30 days after HF diagnosis is treated as prevalent HF)
        days_per_interval = self.survival_days_window / intervals
        survival_then_censor = np.zeros(2*intervals, dtype=np.float32)
        # print(f'Days per interval {days_per_interval} , follow_up_days {follow_up_days} has disease {has_disease} ')
        for i, day_delta in enumerate(np.arange(0, self.survival_days_window, days_per_interval)):
            survival_then_censor[i] = float(day_delta < follow_up_days)
            cur_time_bin = follow_up_days <= day_delta < follow_up_days + days_per_interval
            survival_then_censor[intervals + i] = has_disease * float(cur_time_bin)
            # print(f'day_delta {day_delta} , i {i} has disease {has_disease} cur_time_bin {cur_time_bin}')
            # Handle prevalent diseases
            if has_disease and follow_up_days <= 0:
                survival_then_censor[intervals] = has_disease
        # print(survival_then_censor.shape)
        return survival_then_censor

    @property
    def name(self):
        # if we have multiple wide file DataDescriptions at the same time,
        # this will allow us to differentiate between them
        return self._name
