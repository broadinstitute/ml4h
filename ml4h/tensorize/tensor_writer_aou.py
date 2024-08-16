import os
import h5py
import logging
import numpy as np
import pandas as pd
import argparse

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def assign_paths_to_dfs(file_paths: list) -> tuple:
    """Assign file paths to the appropriate DataFrames based on keywords in the file paths."""
    avg_heart_rate_path = None
    heart_rate_raw_path = None
    diagnosis_path = None
    labs_vitals_path = None
    
    for path in file_paths:
        if 'avg_heart_rate' in path.lower():
            avg_heart_rate_path = path
        elif 'heart_rate_raw' in path.lower():
            heart_rate_raw_path = path
        elif 'diagnosis' in path.lower() or 'condition' in path.lower():
            diagnosis_path = path
        elif 'labs' in path.lower() or 'vitals' in path.lower() or 'measurement' in path.lower():
            labs_vitals_path = path
    
    if not avg_heart_rate_path or not heart_rate_raw_path or not diagnosis_path or not labs_vitals_path:
        raise ValueError("Could not determine one or more file paths from the provided list. Make sure the file paths contain appropriate keywords.")
    
    return avg_heart_rate_path, heart_rate_raw_path, diagnosis_path, labs_vitals_path

def parse_hr_string(hr_string):
    """Parse the string representation of the heart rate array and convert it to a NumPy array."""
    hr_string = hr_string[0].strip('[]').replace('...', '').strip()
    hr_list = hr_string.split()
    hr_array = np.array([float(i) for i in hr_list if i], dtype=np.float32)
    return hr_array

def tensorize_data_per_person(
    avg_heart_rate_df: pd.DataFrame,
    heart_rate_raw_df: pd.DataFrame,
    diagnosis_df: pd.DataFrame,
    labs_vitals_df: pd.DataFrame,
    output_folder: str
) -> None:
    """Create one HDF5 file per person_id with all data stored under date groups."""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Get all unique person_ids
    person_ids = set(avg_heart_rate_df['person_id'].unique()) | set(heart_rate_raw_df['person_id'].unique()) | \
                 set(diagnosis_df['person_id'].unique()) | set(labs_vitals_df['person_id'].unique())

    for person_id in person_ids:
        tensor_path = os.path.join(output_folder, f'{person_id}.h5')
        with h5py.File(tensor_path, 'w') as hdf:
            logging.info(f'Tensorizing data for person_id: {person_id}')
            
            person_avg_hr_data = avg_heart_rate_df[avg_heart_rate_df['person_id'] == person_id]
            person_raw_hr_data = heart_rate_raw_df[heart_rate_raw_df['person_id'] == person_id]
            person_diag_data = diagnosis_df[diagnosis_df['person_id'] == person_id]
            person_labs_data = labs_vitals_df[labs_vitals_df['person_id'] == person_id]
            
            # Get all unique dates
            unique_dates = set(person_avg_hr_data['date'].unique()) | \
                           set(person_raw_hr_data['date'].unique()) | \
                           set(person_diag_data['condition_start_datetime'].dt.date.unique()) | \
                           set(person_labs_data['measurement_datetime'].dt.date.unique())

            for date in unique_dates:
                date_str = str(date)
                date_group = hdf.create_group(date_str)
                
                # Tensorize avg heart rate data for the date
                if date in person_avg_hr_data['date'].unique():
                    avg_hr_data = person_avg_hr_data[person_avg_hr_data['date'] == date]
                    date_group.create_dataset(
                        name='average_heart_rate',
                        data=avg_hr_data[['avg_heart_rate']].to_numpy(),
                        dtype='float32',
                        compression='gzip'
                    )
                
                # Tensorize raw heart rate data for the date
                if date in person_raw_hr_data['date'].unique():
                    raw_hr_data = person_raw_hr_data[person_raw_hr_data['date'] == date]
                    for _, row in raw_hr_data.iterrows():
                        # Convert string to numeric array
                        hr_array = parse_hr_string(row['heart_rate_array'])
                        date_group.create_dataset(
                            name='raw_heart_rate',
                            data=hr_array,
                            dtype='float32',
                            compression='gzip'
                        )
                
                # Tensorize diagnosis data for the date
                diag_data = person_diag_data[person_diag_data['condition_start_datetime'].dt.date == date]
                if not diag_data.empty:
                    date_group.create_dataset(
                        name='condition',
                        data=diag_data[['condition_concept_id', 'standard_concept_name', 'standard_concept_code', 'standard_vocabulary',\
                                        'condition_start_datetime', 'condition_end_datetime', 'condition_type_concept_id',\
                                        'condition_type_concept_name', 'stop_reason', 'visit_occurrence_id', 'visit_occurrence_concept_name',\
                                        'condition_source_value', 'condition_source_concept_id', 'source_concept_name', 'source_concept_code',\
                                        'source_vocabulary', 'condition_status_source_value', 'condition_status_concept_id',\
                                        'condition_status_concept_name']].to_numpy(dtype='S'),
                        dtype='S10',
                        compression='gzip'
                    )

                # Tensorize labs and vitals data for the date
                labs_data = person_labs_data[person_labs_data['measurement_datetime'].dt.date == date]
                if not labs_data.empty:
                    date_group.create_dataset(
                        name='measurement',
                        data=labs_data[['measurement_concept_id', 'standard_concept_name','standard_concept_code','standard_vocabulary',\
                         'measurement_datetime','value_as_number','unit_concept_id','unit_source_value',\
                         'unit_concept_name','visit_occurrence_id','visit_occurrence_concept_name','measurement_source_value',\
                         'measurement_source_concept_id','source_concept_name','source_concept_code','source_vocabulary']].to_numpy(dtype='S'),
                        dtype='S10',
                        compression='gzip'
                    )

        logging.info(f'Data for person_id: {person_id} written to {tensor_path}')


def main(file_paths: list, output_folder: str):
    """Main function to load data, tensorize, and save."""
    # Assign the file paths to the appropriate DataFrames
    avg_heart_rate_path, heart_rate_raw_path, diagnosis_path, labs_vitals_path = assign_paths_to_dfs(file_paths)
    
    avg_heart_rate_df = pd.read_csv(avg_heart_rate_path)
    heart_rate_raw_df = pd.read_csv(heart_rate_raw_path)
    diagnosis_df = pd.read_csv(diagnosis_path)
    labs_vitals_df = pd.read_csv(labs_vitals_path)

    avg_heart_rate_df['date'] = pd.to_datetime(avg_heart_rate_df['date']).dt.date
    heart_rate_raw_df['date'] = pd.to_datetime(heart_rate_raw_df['date']).dt.date
    diagnosis_df['condition_start_datetime'] = pd.to_datetime(diagnosis_df['condition_start_datetime'])
    diagnosis_df['condition_end_datetime'] = pd.to_datetime(diagnosis_df['condition_end_datetime'])
    labs_vitals_df['measurement_datetime'] = pd.to_datetime(labs_vitals_df['measurement_datetime'])

    tensorize_data_per_person(avg_heart_rate_df, heart_rate_raw_df, diagnosis_df, labs_vitals_df, output_folder)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Tensorize data from CSV files into individual HDF5 files per person_id.')
    parser.add_argument('file_paths', nargs='+', help='List of paths to the CSV files (avg_heart_rate, heart_rate_raw, diagnosis, labs_vitals)')
    parser.add_argument('output_folder', type=str, help='Directory to save the output HDF5 files')
    
    args = parser.parse_args()

    main(args.file_paths, args.output_folder)

