import sys
# print("Python executable:", sys.executable)
# print("Environment path:", sys.path)

import dask.dataframe as dd
import pandas as pd
import argparse
import os
import pandas as pd
from tqdm import tqdm

from Merge_India_shape_file_and_make_OD import merge_shape_file, extract_origin_info, district_OD_matrix, district_OD_matrix_datewise
from region_wise_activity_detrmination import extract_region_records, merge_line_shape_file, merge_airport_shape, make_position_groups
from char_dham_activity import merge_char_dham_important_locations, calculate_time_spent, char_dham_visitors, number_of_different_paths_taken, plot_travel_duration, plot_for_multiple_ids, get_activity_subset_counts, calculate_origin_state_counts, plot_heatmap, create_od_matrix, origin_state_origin_district_to_char_dham, find_travellers_source_to_dest, find_return_travellers, plot_time_spent_in_chardham , find_stats_time_spent_in_chardham



def main():
    # Initialize the argument parser
    parser = argparse.ArgumentParser(description='Process and combine CSV files and compute velocities.')

    # Define arguments
    parser.add_argument('--csv', type=str, required=True, help='File path')
    parser.add_argument('--output', type=str, required=True, help='Output directory path for the processed files.')
    parser.add_argument('--India_shape_file', type=str,  required=True, help='path of India_shape_file')
    parser.add_argument('--region_name', type=str, help='region name in short')
    parser.add_argument('--road_shape_file', type=str, help = 'path of road_shape_file LINESHAPE')
    parser.add_argument('--rail_shape_file', type=str, help = 'path of rail_shape_file LINESHAPE')
    
    parser.add_argument('--airport_point_shape_file', type=str, help = 'path of rail_shape_file LINESHAPE')
    parser.add_argument('--airport_boundary_shape_file', type=str, help = 'path of rail_shape_file LINESHAPE')

    # Parse arguments
    args = parser.parse_args()

    if not os.path.exists(args.output):
        os.mkdir(args.output)

    # Combine CSV files
    # print(f"Combining CSV files from directory: {args.csv_dir}")
    df = pd.read_csv(args.csv)


    # Read the CSV file using Dask
    ddf = dd.read_csv(args.csv)
    # print('read')
    # Convert to a pandas DataFrame (if it fits in memory)
    df = ddf.compute()  # This brings the entire dataframe into memory, use only if it can fit
    # print('converted to pandas')
    df = merge_shape_file(df, args.India_shape_file)
    # print('merged')
    df.to_csv(args.output+'\df_with_states_districts.csv', index=False)
    # df.head(10000).to_csv(args.output+'\df_with_states_districts_head_10000.csv', index=False)
    # print('saved')
    df_origin = extract_origin_info(df, maid_column='maid', state_column='state_short', district_column='district')
    # print('extracted origin info')
    df_origin.to_csv(args.output+'\df_origin_info.csv', index=False)
    # df_origin.to_csv(args.output+'\df_origin_info_for_head_10000.csv', index=False)
    # print('saved extracted info')
    # df = pd.read_csv(r"C:\Users\User\Downloads\mobility_traces_uttarakhand\with_shape_files_merged\df_with_states_districts_head_10000.csv")

    

    # Convert string to date format using pandas
    start_date = pd.to_datetime('2022-09-01')
    end_date = pd.to_datetime('2022-09-30')

    # Pass date arguments to the function
    od_matrix = district_OD_matrix(df, start_date, end_date)


    od_matrix.to_csv(args.output+'\df_origin_info_OD_matrix.csv')

    od_matrix_datewise_save_dir = args.output+'\df_origin_info_OD_matrix_datewise'
    district_OD_matrix_datewise(df, start_date, end_date, od_matrix_datewise_save_dir)

    df_extracted  = extract_region_records(df, region_name= args.region_name, column = 'state')

    
    df_extracted.to_csv(args.output+'\df_extracted_'+args.region_name+'.csv', index=False)

    df_extracted = merge_line_shape_file(df_extracted, args.road_shape_file, dist_col='dist_to_road(km)', lat_column='latitude', lon_column='longitude')

    df_extracted.to_csv(args.output+'\df_extracted_'+args.region_name+'_with_roads.csv', index=False)

    df_extracted = merge_line_shape_file(df_extracted, args.rail_shape_file, dist_col='dist_to_rail(km)', lat_column='latitude', lon_column='longitude')

    df_extracted.to_csv(args.output+'\df_extracted_'+args.region_name+'_with_rails.csv', index=False)

    df_extracted = merge_airport_shape(df_extracted, args.airport_point_shape_file,args.airport_boundary_shape_file, dist_col='dist_to_airport(km)', lat_column='latitude', lon_column='longitude')
    df_extracted = merge_line_shape_file(df_extracted, args.rail_shape_file, dist_col='dist_to_rail(km)', lat_column='latitude', lon_column='longitude')
    # # df_extracted 
    # df_extracted = pd.read_csv(args.output+'\df_extracted_with_char_dham_locations.csv')
    df_extracted = make_position_groups(df_extracted, grid_size=250)
    
    df_extracted.to_csv(args.output+'\df_extracted_'+args.region_name+'_with_aiports.csv', index=False)
    # merge_char_dham_important_locations, calculate_time_spent, char_dham_visitors, number_of_different_paths_taken, 
    # plot_travel_duration, plot_for_multiple_ids, 
    # get_activity_subset_counts, calculate_origin_state_counts, plot_heatmap, create_od_matrix, origin_state_origin_district_to_char_dham
    # df_extracted = pd.read_csv(args.output+'\df_extracted_'+args.region_name+'_with_aiports.csv')
    df_extracted = merge_char_dham_important_locations(df_extracted, 'distance_to_location')
    
    
    df_char_dham_visitors = char_dham_visitors(df=df_extracted)

    
    df_char_dham_visitors.to_csv(args.output+'df_char_dham_visitors.csv', index= False)
    df_number_of_different_paths_taken = number_of_different_paths_taken(df=df_extracted)
    fixed_maids = ['000010b7-6bd5-4296-9235-2a824209b773', 'maid2']
    plot_save_path = args.output+'\plots_for_multiple_ids.png'
    plot_for_multiple_ids(df_char_dham_visitors, fixed_maids , plot_save_path)

    df_get_activity_subset_counts = get_activity_subset_counts(df_extracted, activity_column='Chaar_dham_location', group_column='maid')
    df_calculate_origin_state_counts = calculate_origin_state_counts(df_extracted, df_get_activity_subset_counts, activity_col='Chaar_dham_location', origin_state_col='origin_state', maid_col='maid')
    # save_path = args.output+'\heatmap_plot.png'
    # plot_heatmap(df_extracted, save_path , title='Heatmap of Origin State to Char Dham Count', cmap='viridis', figsize=(12, 8))
    df_od_matrix = create_od_matrix(df_extracted, maid_col='maid', district_col='district', activity_col='Chaar_dham_location')
    df_origin_state_origin_district_to_char_dham = origin_state_origin_district_to_char_dham(df_extracted)


    df_number_of_different_paths_taken.to_csv(args.output+'\df_number_of_different_paths_taken.csv', index = False)
    df_get_activity_subset_counts.to_csv(args.output+'\df_get_activity_subset_counts.csv', index = False)
    df_calculate_origin_state_counts.to_csv(args.output+'\df_calculate_origin_state_counts.csv', index = False)
    df_od_matrix.to_csv(args.output+'\df_od_matrix.csv', index = False)
    df_origin_state_origin_district_to_char_dham.to_csv(args.output+'\df_origin_state_origin_district_to_char_dham.csv', index = False)


    travellers_count_summary = pd.DataFrame(columns = ['Source/Passing by', 'Destination', 'Count of travellers',  'Count of return travellers','Count of common IDs', 'Travellers', 'Return travellers', 'Common IDs'])
    # districts
    # char_dhams
        
    districts = df_extracted['first_district'].unique().tolist()
    char_dhams = df_extracted['Chaar_dham_location'].unique().tolist()
    states = df_extracted['origin_state'].unique().tolist()
    index=0
    for dham in char_dhams:
        for dist in districts:
            maids = find_travellers_source_to_dest(df_extracted, dham, dist)
            maids_return = find_return_travellers(df_extracted, dham, dist)
            
            common = list(set(maids) & set(maids_return))
            # Use loc to insert data at the current index
            travellers_count_summary.loc[index] = {
                'Source/Passing by': dist,
                'Destination': dham,
                'Count of travellers': len(maids),
                'Count of return travellers': len(maids_return),
                'Count of common IDs': len(common),
                'Travellers': maids,
                'Return travellers': maids_return,
                'Common IDs': common
            }
            index+=1

    # dist_to_char_dham_df

    # find_stats_source_to_dest(df_chardham_visitors_1, 'Kedarnath', 'NTL', maids)
    start_states = ['MH', 'UP','DL', 'GJ','MP','RJ','WB', 'HR', 'BR', 'CG', 'OD', 'TG']  #major number of visitors



    

    def create_output_folder(path):
        """Create a new folder if it doesn't exist."""
        if not os.path.exists(path):
            os.makedirs(path)

    # Assuming 'args.output' is the base output directory passed as an argument
    output_base_path = args.output

    for dham in char_dhams:
        for dist in start_states:
            # Filter the DataFrame for the specific 'first_district' and 'dham'
            filtered_df = df_origin_state_origin_district_to_char_dham[
                (df_origin_state_origin_district_to_char_dham['first_district'] == dist) & 
                (df_origin_state_origin_district_to_char_dham.columns == dham)  # Assuming dham is a column
            ]

            # Check if the filtered DataFrame is not empty before accessing .iloc[0]
            if not filtered_df.empty:
                maids_travellers = filtered_df[dham].iloc[0]
            else:
                print(f"No data found for district: {dist} and dham: {dham}")
                continue  # Skip this iteration if no data found

            # Define the folder path for the outputs
            output_folder = os.path.join(output_base_path, 'Stats_of_time_spent_at_char_dham_from_major_states')
            create_output_folder(output_folder)  # Create the folder if it doesn't exist

            # Define the save path for the plot
            savepath = os.path.join(output_folder, f'{dist}_to_{dham}.png')

            # Generate stats
            stats_with_outliers, stats_without_outliers = find_stats_time_spent_in_chardham(
                df_extracted, dham, maids_travellers, dist, savepath=savepath)

            # Define paths for saving CSV files
            output_with_outliers = os.path.join(output_folder, f'with_outliers_{dist}_to_{dham}.csv')
            output_without_outliers = os.path.join(output_folder, f'without_outliers_{dist}_to_{dham}.csv')

            # Save the stats as CSV files
            stats_with_outliers.to_csv(output_with_outliers, index=False)
            stats_without_outliers.to_csv(output_without_outliers, index=False)


if __name__ == '__main__':
    main()
