from typing import Dict

import numpy as np
import pandas as pd
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
        return np.squeeze(np.array(data, dtype=np.float32))

    @property
    def name(self):
        # if we have multiple wide file DataDescriptions at the same time,
        # this will allow us to differentiate between them
        return self._name
