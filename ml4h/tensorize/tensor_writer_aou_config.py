import os
import h5py
import logging
import pandas as pd
import numpy as np
import json

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_config(config_file: str) -> dict:
    """Load the configuration file containing DataFrame definitions."""
    with open(config_file, 'r') as file:
        config = json.load(file)
    return config

def find_files_in_folder(folder_path: str, file_pattern: str) -> str:
    """Find a file in the folder that matches the given pattern."""
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file_pattern in file.lower():
                return os.path.join(root, file)
    raise ValueError(f"Could not find file matching pattern: {file_pattern}")

def load_dataframe(file_path: str, date_cols: list = None, sep: str = '\t') -> pd.DataFrame:
    """Load a dataframe from a file, optionally parsing date columns."""
    df = pd.read_csv(file_path, sep=sep)
    if date_cols:
        for col in date_cols:
            df[col] = pd.to_datetime(df[col].apply(lambda x: x[0:10] if not pd.isna(x) else x))
    return df

def _compress_and_save_data(hd5: h5py.Group, name: str, data: np.ndarray, dtype: str = 'float32', compression: str = 'gzip') -> None:
    """Helper function to compress and save data into an HDF5 group."""
    dt = h5py.string_dtype(encoding='utf-8') if dtype == 'str' else dtype
    hd5.create_dataset(name, data=data, dtype=dt, compression=compression)

def tensorize_dataframe(hd5, group_name, df, columns, person_id, date_col, is_array):
    """General function to tensorize a DataFrame and save it into HDF5."""
    person_data = df[df['person_id'] == person_id]
    if not person_data.empty:
        group = hd5.create_group(group_name)
        for date in person_data[date_col].unique():
            date_group = group.create_group(str(date))
            date_data = person_data[person_data[date_col] == date]
            if not date_data.empty:
                for col in columns:
                    if is_array:
                        # Parse the column as an array if needed
                        col_data = np.array([float(x) for x in date_data[col].iloc[0].split(',')])
                        _compress_and_save_data(date_group, col, col_data, dtype='float32')
                    else:
                        col_data = date_data[col].astype(str).to_numpy()
                        _compress_and_save_data(date_group, col, col_data, dtype='str')

def tensorize_data_per_person(dataframes: dict, config: dict, output_folder: str) -> None:
    """Create one HDF5 file per person_id with all data stored under concept-based groups followed by date-based groups."""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    person_ids = set().union(*(df['person_id'].unique() for df in dataframes.values()))
    count = 0

    for person_id in person_ids:
        tensor_path = os.path.join(output_folder, f'{person_id}.hd5')
        with h5py.File(tensor_path, 'w') as hd5:
            logging.info(f'Tensorizing data for person_id: {person_id}')
            
            for df_key, config_data in config['dataframes'].items():
                df = dataframes.get(df_key)
                tensorize_dataframe(hd5, df_key, df, config_data['columns'], person_id, config_data['date_columns'][0], config_data['is_array'])

        logging.info(f'Data for person_id: {person_id} written to {tensor_path}')
        count += 1  # Increment the counter
        if count % 100 == 0:
            print(f"{count} person_ids processed.")

def main(folder_path: str, output_folder: str, config_file: str):
    """Main function to load data, tensorize, and save."""
    config = load_config(config_file)
    
    # Find and load all dataframes
    dataframes = {}
    for df_key, df_info in config['dataframes'].items():
        file_path = find_files_in_folder(folder_path, df_info['file_pattern'])
        dataframes[df_key] = load_dataframe(file_path, df_info['date_columns'])

    # Tensorize data per person
    tensorize_data_per_person(dataframes, config, output_folder)

if __name__ == '__main__':
    folder_path = '/path/to/your/folder'  # Replace with your folder path
    output_folder = '/path/to/output/folder'  # Replace with your output folder path
    config_file = 'config.json'

    main(folder_path=folder_path, output_folder=output_folder, config_file=config_file)

