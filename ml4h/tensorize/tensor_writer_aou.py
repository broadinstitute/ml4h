import os
import h5py
import logging
import pandas as pd
import numpy as np
import argparse

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def assign_paths_to_dfs(file_paths: list) -> dict:
    """Assign file paths to the appropriate DataFrames based on keywords in the file paths."""
    paths = {
        'avg_heart_rate_path': None,
        'heart_rate_raw_path': None,
        'diagnosis_path': None,
        'labs_vitals_path': None,
        'sleep_summary_path': None,
        'intraday_steps_path': None,
        'sleep_level_path': None,
        'procedure_path': None,
        'heart_rate_summary_path': None,
        'fitbit_activity_path': None
    }

    keywords = {
        'avg_heart_rate_path': 'fitbit_heart_rate_level_avg',
        'heart_rate_raw_path': 'heart_rate_level',
        'diagnosis_path': ['diagnosis', 'condition'],
        'labs_vitals_path': ['labs', 'vitals', 'measurement'],
        'sleep_summary_path': 'sleep_daily_summary',
        'intraday_steps_path': 'intraday_steps',
        'sleep_level_path': 'sleep_level',
        'procedure_path': 'procedure',
        'heart_rate_summary_path': 'heart_rate_summary',
        'fitbit_activity_path': 'fitbit_activity'
    }

    for path in file_paths:
        for key, kw in keywords.items():
            if isinstance(kw, list):
                if any(k in path.lower() for k in kw):
                    paths[key] = path
            elif kw in path.lower():
                paths[key] = path
    
    if not all(paths.values()):
        raise ValueError("Could not determine one or more file paths from the provided list. Make sure the file paths contain appropriate keywords.")
    
    return paths

def _compress_and_save_data(hd5: h5py.Group, name: str, data: np.ndarray, dtype: str = 'float32', compression: str = 'gzip') -> None:
    """Helper function to compress and save data into an HDF5 group."""
    dt = h5py.string_dtype(encoding='utf-8') if dtype == 'str' else dtype
    hd5.create_dataset(name, data=data, dtype=dt, compression=compression)

def parse_heart_rate_string(hr_string: str) -> np.ndarray:
    """Parse the string representation of the heart rate array and convert it to a NumPy array."""
    hr_list = hr_string.split(',')
    return np.array([float(i) for i in hr_list if i], dtype=np.float32)

def tensorize_dataframe(hd5, group_name, df, columns, person_id, date_col):
    """General function to tensorize a DataFrame and save it into HDF5."""
    person_data = df[df['person_id'] == person_id]
    if not person_data.empty:
        group = hd5.create_group(group_name)
        for date in person_data[date_col].unique():
            date_group = group.create_group(str(date))
            date_data = person_data[person_data[date_col] == date]
            if not date_data.empty:
                for col in columns:
                    col_data = date_data[col].astype(str).to_numpy()
                    _compress_and_save_data(date_group, col, col_data, dtype='str')

def tensorize_data_per_person(
    avg_heart_rate_df: pd.DataFrame,
    heart_rate_raw_df: pd.DataFrame,
    diagnosis_df: pd.DataFrame,
    labs_vitals_df: pd.DataFrame,
    sleep_summary_df: pd.DataFrame,
    intraday_steps_df: pd.DataFrame,
    sleep_level_df: pd.DataFrame,
    procedure_df: pd.DataFrame,
    heart_rate_summary_df: pd.DataFrame,
    fitbit_activity_df: pd.DataFrame,
    output_folder: str
) -> None:
    """Create one HDF5 file per person_id with all data stored under concept-based groups followed by date-based groups."""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    person_ids = set(avg_heart_rate_df['person_id'].unique()) | \
                 set(heart_rate_raw_df['person_id'].unique()) | \
                 set(diagnosis_df['person_id'].unique()) | \
                 set(labs_vitals_df['person_id'].unique()) | \
                 set(sleep_summary_df['person_id'].unique()) | \
                 set(intraday_steps_df['person_id'].unique()) | \
                 set(sleep_level_df['person_id'].unique()) | \
                 set(procedure_df['person_id'].unique()) | \
                 set(heart_rate_summary_df['person_id'].unique()) | \
                 set(fitbit_activity_df['person_id'].unique())

    for person_id in person_ids:
        tensor_path = os.path.join(output_folder, f'{person_id}.hd5')
        with h5py.File(tensor_path, 'w') as hd5:
            logging.info(f'Tensorizing data for person_id: {person_id}')

            # Avg Heart Rate
            tensorize_dataframe(hd5, 'avg_heart_rate', avg_heart_rate_df, ['avg_rate'], person_id, 'date')

            # Raw Heart Rate
            person_hr_raw_data = heart_rate_raw_df[heart_rate_raw_df['person_id'] == person_id]
            if not person_hr_raw_data.empty:
                raw_hr_group = hd5.create_group('raw_heart_rate')
                for date in person_hr_raw_data['date'].unique():
                    date_group = raw_hr_group.create_group(str(date))
                    hr_raw_data = person_hr_raw_data[person_hr_raw_data['date'] == date]
                    if not hr_raw_data.empty:
                        for _, row in hr_raw_data.iterrows():
                            hr_array = parse_heart_rate_string(row['heart_rate_array'])
                            _compress_and_save_data(date_group, 'raw_heart_rate', hr_array)

            # Sleep Summary
            tensorize_dataframe(hd5, 'sleep_summary', sleep_summary_df, [
                'is_main_sleep', 'minute_in_bed', 'minute_asleep', 'minute_after_wakeup',
                'minute_awake', 'minute_restless', 'minute_deep', 'minute_light',
                'minute_rem', 'minute_wake'], person_id, 'sleep_date')

            # Intraday Steps
            tensorize_dataframe(hd5, 'intraday_steps', intraday_steps_df, ['sum_steps'], person_id, 'date')

            # Sleep Level
            tensorize_dataframe(hd5, 'sleep_level', sleep_level_df, [
                'is_main_sleep', 'level', 'date', 'duration_in_min'], person_id, 'sleep_date')

            # Diagnosis
            tensorize_dataframe(hd5, 'condition', diagnosis_df, [
                'condition_concept_id', 'standard_concept_name', 'standard_concept_code',
                'standard_vocabulary', 'condition_start_datetime', 'condition_end_datetime',
                'condition_type_concept_id', 'condition_type_concept_name', 'stop_reason',
                'visit_occurrence_id', 'visit_occurrence_concept_name', 'condition_source_value',
                'condition_source_concept_id', 'source_concept_name', 'source_concept_code',
                'source_vocabulary', 'condition_status_source_value', 'condition_status_concept_id',
                'condition_status_concept_name'], person_id, 'condition_start_datetime')

            # Labs and Vitals
            tensorize_dataframe(hd5, 'measurement', labs_vitals_df, [
                'measurement_concept_id', 'standard_concept_name', 'standard_concept_code', 'standard_vocabulary',
                'value_as_number', 'unit_concept_id', 'unit_source_value',
                'unit_concept_name', 'visit_occurrence_id', 'visit_occurrence_concept_name', 'measurement_source_value',
                'measurement_source_concept_id', 'source_concept_name', 'source_concept_code', 'source_vocabulary'
            ], person_id, 'measurement_datetime')

            # Procedure
            tensorize_dataframe(hd5, 'procedure', procedure_df, [
                'procedure_concept_id', 'standard_concept_name', 'standard_concept_code', 'standard_vocabulary',
                'procedure_type_concept_id', 'procedure_type_concept_name', 'modifier_concept_id', 'modifier_concept_name',
                'quantity', 'visit_occurrence_id', 'visit_occurrence_concept_name', 'procedure_source_value',
                'procedure_source_concept_id', 'source_concept_name', 'source_concept_code', 'source_vocabulary',
                'modifier_source_value'
            ], person_id, 'procedure_datetime')

            # Heart Rate Summary
            tensorize_dataframe(hd5, 'heart_rate_summary', heart_rate_summary_df, [
                'zone_name', 'min_heart_rate', 'max_heart_rate', 'minute_in_zone', 'calorie_count'
            ], person_id, 'date')

            # Fitbit Activity
            tensorize_dataframe(hd5, 'fitbit_activity', fitbit_activity_df, [
                'activity_calories', 'calories_bmr', 'calories_out', 'elevation', 'fairly_active_minutes', 'floors',
                'lightly_active_minutes', 'marginal_calories', 'sedentary_minutes', 'steps', 'very_active_minutes'
            ], person_id, 'date')

        logging.info(f'Data for person_id: {person_id} written to {tensor_path}')



def load_dataframe(file_path: str, date_cols: list = None, sep: str = '\t') -> pd.DataFrame:
    """Load a dataframe from a file, optionally parsing date columns."""
    df = pd.read_csv(file_path, sep=sep)
    if date_cols:
        for col in date_cols:
            df[col] = pd.to_datetime(df[col].apply(lambda x: x[0:10] if not pd.isna(x) else x))
    return df

def main(file_paths: list, output_folder: str):
    """Main function to load data, tensorize, and save."""
    paths = assign_paths_to_dfs(file_paths)
    
    # Load all dataframes with corresponding date columns
    dataframes = {
        'avg_heart_rate_df': load_dataframe(paths['avg_heart_rate_path'], ['date']),
        'heart_rate_raw_df': load_dataframe(paths['heart_rate_raw_path'], ['date']),
        'sleep_summary_df': load_dataframe(paths['sleep_summary_path'], ['sleep_date']),
        'diagnosis_df': load_dataframe(paths['diagnosis_path'], ['condition_start_datetime', 'condition_end_datetime']),
        'labs_vitals_df': load_dataframe(paths['labs_vitals_path'], ['measurement_datetime']),
        'intraday_steps_df': load_dataframe(paths['intraday_steps_path'], ['date']),
        'sleep_level_df': load_dataframe(paths['sleep_level_path'], ['sleep_date']),
        'procedure_df': load_dataframe(paths['procedure_path'], ['procedure_datetime']),
        'heart_rate_summary_df': load_dataframe(paths['heart_rate_summary_path'], ['date']),
        'fitbit_activity_df': load_dataframe(paths['fitbit_activity_path'], ['date'])
    }

    # Tensorize data per person
    tensorize_data_per_person(
        avg_heart_rate_df=dataframes['avg_heart_rate_df'], 
        heart_rate_raw_df=dataframes['heart_rate_raw_df'], 
        diagnosis_df=dataframes['diagnosis_df'], 
        labs_vitals_df=dataframes['labs_vitals_df'], 
        sleep_summary_df=dataframes['sleep_summary_df'], 
        intraday_steps_df=dataframes['intraday_steps_df'], 
        sleep_level_df=dataframes['sleep_level_df'], 
        procedure_df=dataframes['procedure_df'], 
        heart_rate_summary_df=dataframes['heart_rate_summary_df'],
        fitbit_activity_df=dataframes['fitbit_activity_df'],
        output_folder=output_folder
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load data, tensorize, and save.")
    parser.add_argument(
        'file_paths', 
        nargs='+', 
        help='List of file paths to be processed, including fitbit, diagnosis, labs, etc.'
    )
    parser.add_argument(
        '--output_folder', 
        required=True, 
        help='Directory to save the output HDF5 files.'
    )
    args = parser.parse_args()
    
    main(file_paths=args.file_paths, output_folder=args.output_folder)
