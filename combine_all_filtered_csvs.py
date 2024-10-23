import pandas as pd
import os
from tqdm import tqdm
import warnings
import sys
import gc
import dask.dataframe as dd

# Suppress FutureWarnings
warnings.filterwarnings("ignore", category=FutureWarning)

def process_csv_files(csv_path, output_path, columns_to_extract=None):
    """
    Function to process CSV files from the given directory, clean, sort, and save the output to the specified path.
    
    Parameters:
    - csv_path (str): The directory path where the CSV files are stored.
    - output_path (str): The directory path where the final output CSV file will be saved.
    - columns_to_extract (list): List of column names to extract from the CSV files. If None, default columns will be used.
    """
    if columns_to_extract is None:
        columns_to_extract = ['maid', 'latitude', 'longitude', 'datetime', 'ACCURACY']
    
    # List CSV files in the directory
    csv_files = [file for file in os.listdir(csv_path) if file.endswith('.csv')]
    # Initialize a list to collect DataFrames
    dfs = []
    count = 0

    final_output_path = os.path.join(output_path, 'df_all_without_dups_sorted.csv')
    print(f' final_output_path: {final_output_path}')
    # Iterate over CSV files with a progress bar
    
    for csv_file in tqdm(csv_files, desc='Processing Files', leave=False, ncols=80, ascii=True, file=sys.stdout):
        csv_file_path = os.path.join(csv_path, csv_file)
        try:
            # Read each CSV file into a DataFrame with specified columns
            df_file = pd.read_csv(csv_file_path, usecols=columns_to_extract)
            df_file['latitude'] = df_file['latitude'].astype(float).round(4)
            df_file['longitude'] = df_file['longitude'].astype(float).round(4)
            df_file = df_file.drop_duplicates(subset=['maid', 'datetime', 'latitude', 'longitude'], keep='first')
            
            # Append the cleaned DataFrame to the list
            dfs.append(df_file)
            count += len(df_file)
        except Exception as e:
            print(f"Error reading {csv_file}: {e}")

    # Concatenate all DataFrames in the list
    if dfs:
        df_all = pd.concat(dfs, ignore_index=True)
    else:
        print("No CSV files to process.")
        return

    # Free memory
    del dfs
    gc.collect()

    # Data cleaning and memory optimization
    df_all['latitude'] = df_all['latitude'].astype(float).round(4)
    df_all['ACCURACY'] = df_all['ACCURACY'].astype(float).round(2)

    # Save the concatenated DataFrame before sorting
    intermediate_path = os.path.join(output_path, 'df_all_filtered_records.csv')
    df_all.to_csv(intermediate_path, index=False)
    print(f"Intermediate CSV saved at {intermediate_path}")

    # Convert pandas DataFrame to Dask DataFrame for sorting
    ddf = dd.from_pandas(df_all, npartitions=8)  # Adjust partitions as needed

    # Sort by the specified columns using Dask
    ddf_sorted = ddf.sort_values(by=['maid', 'datetime', 'latitude', 'longitude'])

    # Compute the sorted Dask DataFrame back into pandas
    df_all = ddf_sorted.compute()
    print(f"DataFrame shape after sorting: {df_all.shape}")

    # Drop duplicates and unwanted columns if they exist
    df_all = df_all.drop_duplicates(subset=['maid', 'datetime', 'latitude', 'longitude'], keep='first')
    # df_all.drop(columns=['level_0', 'index'], inplace=True, errors='ignore')
    print(f"DataFrame shape after dropping duplicates: {df_all.shape}")

    # # Save the final sorted and cleaned DataFrame
    # final_output_path = os.path.join(output_path, 'df_all_without_dups_sorted.csv')
    df_all.to_csv(final_output_path, index=False)
    print(f"Final CSV saved at {final_output_path}")

    # Return the final DataFrame for further use
    return df_all