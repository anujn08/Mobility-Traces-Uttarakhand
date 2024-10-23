import pandas as pd
import argparse
from dist_velocity_type_1_type_2_computation import optimized_function_maid_vectorized, remove_type_1, remove_type_2, remove_type_3, \
                                                find_airports, convert_to_CRS
from combine_all_filtered_csvs import process_csv_files
from remove_unnecessary_records import remove_unnecessary_timestamps, filter_main_dataframe_by_conditions
import os
import dask.dataframe as dd
import pandas as pd
from tqdm import tqdm

def main():
    # Initialize the argument parser
    parser = argparse.ArgumentParser(description='Process and combine CSV files and compute velocities.')

    # Define arguments
    parser.add_argument('--csv_dir', type=str, required=True, help='Directory containing CSV files to combine.')
    parser.add_argument('--output_csv', type=str, required=True, help='Output file path for the combined CSV.')
    parser.add_argument('--type_1_save_path', type=str,  required=True, help='output files path for type 1')
    parser.add_argument('--type_2_save_path', type=str, help='output files path for type 2')

    # Parse arguments
    args = parser.parse_args()

    # Combine CSV files
    # print(f"Combining CSV files from directory: {args.csv_dir}")
    # df_all = process_csv_files(args.csv_dir, args.output_csv)
    # df_all["y"] ,df_all["x"] = convert_to_CRS(df_all.latitude , df_all.longitude  )
    # df_all = optimized_function_maid_vectorized(df_all, lat_col='latitude', lon_col='longitude', datetime_col='datetime', maid_col='maid')
    # os.makedirs(args.output_csv)
    # df_all.to_csv(args.output_csv+'\df_all_with_distance.csv', index=False)
    # df_all = pd.read_csv(r"C:\Users\User\Downloads\Mobility traces Mumbai\type_2_removal\df_without_type_1_step_16.csv")
    # df_all = pd.read_csv(args.output_csv+'\df_all_with_distance.csv')
    # df_all= remove_type_1(df_all, args.type_1_save_path)
    # df_all= remove_type_2(df_all, args.type_2_save_path)
    # find_airports(df_all)
    # df_all = remove_type_3(df_all)  # default arguments: velocity_threshold=150, displacement_threshold=150, min_velocity=60, max_velocity=90
    # df_all = df_all[['maid', 'latitude', 'longitude', 'datetime', 'ACCURACY', 'y', 'x',
    #    'displacement', 'distance', 'Velocity',  'osm_id', 'name_en']]
    # df_all =pd.read_csv(args.output_csv+'\df_without_type_3_new.csv')
    # print(df_all.shape)
    # df_all = remove_unnecessary_timestamps(df_all)
    # print(df_all.shape)
    # df_all.to_csv(args.output_csv+'\df_all_with_fixed_anolamies_without_unnecessary_records.csv', index=False)
    
    df_all = pd.read_csv(args.output_csv+'\df_all_with_fixed_anolamies_without_unnecessary_records.csv')
    # print(df_all.head())
    df_all = filter_main_dataframe_by_conditions(df_all, min_points=4, continuous_days=3)
    df_all.to_csv(args.output_csv+'\df_all_valid_distributed_IDs_data.csv')






if __name__ == '__main__':
    main()
