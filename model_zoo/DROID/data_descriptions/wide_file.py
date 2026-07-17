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
    def _survival_tensor_from_row(row, config):
        """Encode one event/censoring observation for discrete survival loss.

        The first half marks intervals that were survived.  The second half is
        one-hot only when an event occurs within the prediction horizon.  The
        corresponding model head predicts one conditional-survival probability
        per interval, so its width is ``intervals`` rather than ``2 * intervals``.
        """
        intervals = config['intervals']
        days_window = config['days_window']
        has_event = float(row[config['event_column']]) == 1.0
        follow_up_days = float(row[config['follow_up_days_column']]) - config['blanking_days']
        days_per_interval = days_window / intervals
        target = np.zeros(2 * intervals, dtype=np.float32)

        if has_event and follow_up_days <= 0:
            # A prevalent event is represented as a failure in the first bin.
            target[intervals] = 1.0
            return target

        for interval, interval_start in enumerate(np.arange(0, days_window, days_per_interval)):
            interval_end = interval_start + days_per_interval
            # Only intervals completed before the event/censoring time count as
            # survived.  The event bin is represented solely in the second half.
            target[interval] = float(interval_end <= follow_up_days)
            if has_event and interval_start <= follow_up_days < interval_end:
                target[intervals + interval] = 1.0
        return target

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
        # ---------- Adaptation for regression + classification + survival ---------- #
        if self.cls_categories_map or self.survival_task_configs:
            data = []
            if self.column_names:
                regression_columns = self.column_names
            else:
                regression_columns = []
            if self.cls_categories_map:
                regression_columns = [
                    column for column in regression_columns
                    if column not in self.cls_categories_map['cls_output_order']
                ]
            reg_data = row[regression_columns].values
            if len(reg_data) > 0:
                data.append(np.squeeze(np.array(reg_data, dtype=np.float32)))

            if self.cls_categories_map:
                for k in self.cls_categories_map['cls_output_order']:
                    # Changing values to class labels:
                    row_cls_lbl = self.cls_categories_map[k][row[k]]
                    # Changing class indices to one hot vectors
                    cls_one_hot = tf.keras.utils.to_categorical(row_cls_lbl,
                                                                num_classes=len(self.cls_categories_map[k]))
                    data.append(cls_one_hot)

            for config in self.survival_task_configs:
                data.append(self._survival_tensor_from_row(row, config))

            if len(data) == 1:
                data = data[0]

            return data
        # ---------------------------------------------------------------- #
        return np.squeeze(np.array(data, dtype=np.float32))

    @property
    def name(self):
        # if we have multiple wide file DataDescriptions at the same time,
        # this will allow us to differentiate between them
        return self._name
