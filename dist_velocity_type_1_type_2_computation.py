import pandas as pd
import numpy as np
from pyproj import CRS, Transformer
# from geopy.distance import geodesic
import warnings
import json
import os
from tqdm import tqdm
    
warnings.filterwarnings("ignore")

# Projection
crs_4326 = CRS("WGS84")  # source CRS that is lat and lon
crs_proj = CRS("EPSG:3857")  # for Uttarakhand, x and y
transformer = Transformer.from_crs(crs_4326, crs_proj)

def convert_to_CRS(latitude, longitude):
    y, x = transformer.transform(latitude, longitude)
    return (y, x)

def euclidean(coord1, coord2):
    """
    Calculate the Euclidean distance between two points given their coordinates.

    Parameters:
    coord1 (tuple): A tuple (x1, y1) representing the coordinates of the first point.
    coord2 (tuple): A tuple (x2, y2) representing the coordinates of the second point.

    Returns:
    float: The Euclidean distance between the two points.
    """
    x1, y1 = coord1
    x2, y2 = coord2
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)



def calculate_distances_and_time_gaps_continous_invalid_stamps(data, displacement_threshold, velocity_threshold, time_window, distance_range):
    """
    Calculate distances and time gaps between consecutive records for each 'maid' and flag invalid stamps.

    Parameters:
    - data (pd.DataFrame): Input DataFrame with 'latitude', 'longitude', 'datetime', 'maid', and 'displacement' columns.
    - displacement_threshold (float): Minimum displacement threshold to consider as invalid.
    - velocity_threshold (float): Minimum velocity threshold to consider as invalid.
    - time_window (float): Maximum time gap (in minutes) between valid consecutive records.
    - distance_range (float): Maximum distance range (in kilometers) between consecutive records.

    Returns:
    - pd.DataFrame: Updated DataFrame with new columns for distance, time gap, and validity flag.
    """

    # Convert 'datetime' column to datetime format
    data['datetime'] = pd.to_datetime(data['datetime'])
    # Initialize new columns
    data['dist_previous_next'] = None
    data['time_gap_previous_next'] = None
    data['next_valid_index'] = None
    data['keep'] = True

    # Sort data by 'maid', 'datetime', 'latitude', and 'longitude', and reset index
    data = data.sort_values(by=['maid', 'datetime', 'latitude', 'longitude']).reset_index(drop=True)

    # Maintain a mapping from the original index to the new sorted index
    original_to_sorted_index = data.reset_index().set_index('index').index
    # Initialize the progress bar
    tqdm.pandas(desc="Processing maids")

    # Calculate distance and time gap
    for maid, group in tqdm(data.groupby('maid'), desc="Processing each maid"):
        num_records = len(group)
        for i in range(1, num_records - 1):
            displacement = group.loc[group.index[i], 'displacement']
            velocity = group.loc[group.index[i], 'Velocity']
            if (displacement >= displacement_threshold) and (velocity >= velocity_threshold):
                prev_record = group.loc[group.index[i - 1]]
                next_valid_index = None

                for j in range(i + 1, num_records):
                    next_record = group.loc[group.index[j]]
                    time_gap = (next_record['datetime'] - prev_record['datetime']).total_seconds() / 60

                    if time_gap > time_window:
                        break

                    # distance = geodesic(
                    #     (prev_record['latitude'], prev_record['longitude']),
                    #     (next_record['latitude'], next_record['longitude'])
                    # ).kilometers

                    distance = euclidean(
                        (prev_record['x'], prev_record['y']),
                        (next_record['x'], next_record['y'])
                    )/1000  # kilometers

                    if distance < distance_range:
                        next_valid_index = j
                        break

                if next_valid_index is not None:
                    original_index = group.index[i]
                    sorted_next_valid_index = group.index[next_valid_index]

                    # Convert sorted index back to original index
                    original_next_valid_index = original_to_sorted_index.get_loc(sorted_next_valid_index)

                    # Update the current record (i) with information about the next valid index
                    data.at[original_index, 'next_valid_index'] = original_next_valid_index
                    data.at[original_index, 'dist_previous_next'] = distance
                    data.at[original_index, 'time_gap_previous_next'] = time_gap

                    # Mark records between the current index and one record before the next valid index as False
                    if original_next_valid_index > original_index:
                        indices_to_mark = range(original_index, original_next_valid_index)
                        data.loc[indices_to_mark, 'keep'] = False

    return data



def calculate_distances_and_time_gaps_continous_invalid_stamps_looking_backward(data, displacement_threshold, velocity_threshold, time_window, distance_range):
    # Convert 'datetime' column to datetime format
    data['datetime'] = pd.to_datetime(data['datetime'])
    # col_name = f'keep_for_{displacement_threshold}_{velocity_threshold}_{time_window}_{distance_range}'
    # Initialize new columns
    data['dist_previous_next'] = None
    data['time_gap_previous_next'] = None
    data['prev_valid_index'] = None
    data['keep'] = True
    
    # Sort data by 'maid', 'datetime', 'latitude', and 'longitude', and reset index
    data = data.sort_values(by=['maid', 'datetime', 'latitude', 'longitude']).reset_index(drop=True)
    
    # Maintain a mapping from the original index to the new sorted index
    original_to_sorted_index = data.reset_index().set_index('index').index
    # Initialize the progress bar
    tqdm.pandas(desc="Processing maids")

    # Calculate distance and time gap
    for maid, group in tqdm(data.groupby('maid'), desc="Processing each maid"):
    # Calculate distance and time gap
    # for maid, group in data.groupby('maid'):
        num_records = len(group)
        for i in range(1, num_records - 1):
        # for i in range(1, num_records - 1):
            displacement = group.loc[group.index[i], 'displacement']
            # print(f'group.index[i]: {group.index[i]}')
            # print(f'disalcement:{displacement}')
            velocity = group.loc[group.index[i], 'Velocity']
            # print(f'disalcement:{displacement}, velocity { {velocity}}')
            if (displacement >= displacement_threshold) and (velocity >= velocity_threshold):
                # print(f'Inside if: disalcement:{displacement}, velocity { {velocity}}')
                # print(f'index: {group.index[i]}')
                curr_record = group.loc[group.index[i]]
                prev_valid_index = None
                
                for j in range(i - 1, 0,-1):
                    prev_record = group.loc[group.index[j]]
                    # print(f'prev_record= {prev_record}')
                    # print(f'index of prev_record: {j}')
                    time_gap = (curr_record['datetime'] - prev_record['datetime']).total_seconds() / 60
                    # print(f'time_gap:{time_gap} {type(time_gap)}')
                    # print(f'time_window: {time_window} {type(time_window)}')
                    if time_gap > time_window:
                        
                        # print('break due to time gap')
                        break
                    
                    # distance = geodesic(
                    #     (prev_record['latitude'], prev_record['longitude']),
                    #     (curr_record['latitude'], curr_record['longitude'])
                    # ).kilometers
                    distance = euclidean(
                        (prev_record['x'], prev_record['y']),
                        (curr_record['x'], curr_record['y'])
                    )/1000


                    if distance < distance_range:
                        prev_valid_index = j
                        # print('break due to distance')
                        break

                if prev_valid_index is not None:
                    original_index = group.index[i]
                    sorted_previous_valid_index = group.index[prev_valid_index]

                    # Convert sorted index back to original index
                    original_previous_valid_index = original_to_sorted_index.get_loc(sorted_previous_valid_index)
                    # print(f'original_index: {original_index}')
                    # print(f'original_previous_valid_index: {original_previous_valid_index}')
                    # Update the current record (i) with information about the next valid index
                    data.at[original_index, 'prev_valid_index'] = original_previous_valid_index
                    data.at[original_index, 'dist_previous_next'] = distance
                    data.at[original_index, 'time_gap_previous_next'] = time_gap

                    # Mark records between the one record previous to current index and one record later than the previous valid index as False
                    if original_previous_valid_index < original_index:
                        indices_to_mark = range(original_previous_valid_index + 1, original_index )
                        # print(f'indices_to_mark: {indices_to_mark}')
                        data.loc[indices_to_mark, 'keep'] = False

    return data



def optimized_function_maid_vectorized(df, lat_col='latitude', lon_col='longitude', datetime_col='datetime', maid_col='maid'):
    """
    Optimize DataFrame by computing distances and velocities using vectorized operations.

    Parameters:
    - df (pd.DataFrame): Input DataFrame with 'latitude', 'longitude', and 'datetime' columns.
    - lat_col (str): Name of the column containing latitude values.
    - lon_col (str): Name of the column containing longitude values.
    - datetime_col (str): Name of the column containing datetime values.
    - maid_col (str): Name of the column containing maid values.

    Returns:
    - pd.DataFrame: DataFrame with computed distances, velocities, and time gaps.
    """
    df['datetime'] = pd.to_datetime(df[datetime_col])
    df = df.sort_values(by=[maid_col, datetime_col, lat_col, lon_col])

    # Shift columns to get previous values for each maid
    df['prev_X'] = df.groupby(maid_col)[lon_col].shift(1)
    df['prev_Y'] = df.groupby(maid_col)[lat_col].shift(1)
    df['prev_datetime'] = df.groupby(maid_col)[datetime_col].shift(1)

    # Radius of the Earth in kilometers
    R = 6371

    # Convert latitude and longitude from degrees to radians
    lat1 = np.radians(df[lat_col].values)
    lat2 = np.radians(df['prev_Y'].values)
    dlat = lat2 - lat1
    dlon = np.radians(df['prev_X'].values) - np.radians(df[lon_col].values)

    # Haversine formula for distance calculation
    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    # Calculate consecutive distance in kilometers
    df['displacement'] = R * c
    df['distance'] = df.groupby(maid_col)['displacement'].cumsum()

    # Calculate consecutive time gap in minutes
    df['consecutive_time_gap(minutes)'] = (df[datetime_col] - df['prev_datetime']).dt.total_seconds() / 60

    # Calculate velocity
    df['Velocity'] = df['displacement'] / (df['consecutive_time_gap(minutes)'] / 60)

    # Drop temporary columns
    df = df.drop(columns=['prev_X', 'prev_Y', 'prev_datetime', 'consecutive_time_gap(minutes)'])

    # Fill NaN values with 0
    df = df.fillna(0)

    return df


def filter_out_continuos_invalid_stamps(data):
    filtered_data = data[data['keep'] == True]
    return filtered_data


def params(step):

    # determine parameters:
    if step==1:
        displacement_min_threshold = 75   # distance covered during invalid transition
        min_velocity_threshold = 150       # minimum speed during the invalid transition
        dist_prev_next_threshold = 50      # maximum permittted distance between previos and next location for the invalid transition stamp
        time_gap_pre_next_threshold = 60 # Time gap(min) between previuos and next time stamp for the invalid record
        sub_folder_name = f'threshold_dist_{displacement_min_threshold}_vel_{min_velocity_threshold}_dist_prev_next_{dist_prev_next_threshold}_time_gap_{time_gap_pre_next_threshold}_min'
    elif step==2:
        displacement_min_threshold = 150   # distance covered during invalid transition
        min_velocity_threshold = 150       # minimum speed during the invalid transition
        dist_prev_next_threshold = 100      # maximum permittted distance between previos and next location for the invalid transition stamp
        time_gap_pre_next_threshold = 120 # Time gap(min) between previuos and next time stamp for the invalid record
        sub_folder_name = f'threshold_dist_{displacement_min_threshold}_vel_{min_velocity_threshold}_dist_prev_next_{dist_prev_next_threshold}_time_gap_{time_gap_pre_next_threshold}_min'
    elif step==3:
        displacement_min_threshold = 225   # distance covered during invalid transition
        min_velocity_threshold = 150       # minimum speed during the invalid transition
        dist_prev_next_threshold = 150      # maximum permittted distance between previos and next location for the invalid transition stamp
        time_gap_pre_next_threshold = 180 # Time gap(min) between previuos and next time stamp for the invalid record
        sub_folder_name = f'threshold_dist_{displacement_min_threshold}_vel_{min_velocity_threshold}_dist_prev_next_{dist_prev_next_threshold}_time_gap_{time_gap_pre_next_threshold}_min'
    elif step ==4:
        displacement_min_threshold = 300   # distance covered during invalid transition
        min_velocity_threshold = 150       # minimum speed during the invalid transition
        dist_prev_next_threshold = 200      # maximum permittted distance between previos and next location for the invalid transition stamp
        time_gap_pre_next_threshold = 240 # Time gap(min) between previuos and next time stamp for the invalid record
        sub_folder_name = f'threshold_dist_{displacement_min_threshold}_vel_{min_velocity_threshold}_dist_prev_next_{dist_prev_next_threshold}_time_gap_{time_gap_pre_next_threshold}_min'
    elif step==5:
        displacement_min_threshold = 375   # distance covered during invalid transition
        min_velocity_threshold = 150       # minimum speed during the invalid transition
        dist_prev_next_threshold = 250      # maximum permittted distance between previos and next location for the invalid transition stamp
        time_gap_pre_next_threshold = 300 # Time gap(min) between previuos and next time stamp for the invalid record
        sub_folder_name = f'threshold_dist_{displacement_min_threshold}_vel_{min_velocity_threshold}_dist_prev_next_{dist_prev_next_threshold}_time_gap_{time_gap_pre_next_threshold}_min'
    elif step==6:
        displacement_min_threshold = 300   # distance covered during invalid transition
        min_velocity_threshold = 150       # minimum speed during the invalid transition
        dist_prev_next_threshold = 300      # maximum permittted distance between previos and next location for the invalid transition stamp
        time_gap_pre_next_threshold = 360 # Time gap(min) between previuos and next time stamp for the invalid record
        sub_folder_name = f'threshold_dist_{displacement_min_threshold}_vel_{min_velocity_threshold}_dist_prev_next_{dist_prev_next_threshold}_time_gap_{time_gap_pre_next_threshold}_min'
    elif step==7:
        displacement_min_threshold = 700   # distance covered during invalid transition
        min_velocity_threshold = 500      # minimum speed during the invalid transition
        dist_prev_next_threshold = 350      # maximum permittted distance between previos and next location for the invalid transition stamp
        time_gap_pre_next_threshold = 420 # Time gap(min) between previuos and next time stamp for the invalid record
        sub_folder_name = f'threshold_dist_{displacement_min_threshold}_vel_{min_velocity_threshold}_dist_prev_next_{dist_prev_next_threshold}_time_gap_{time_gap_pre_next_threshold}_min'
    elif step==8:
        displacement_min_threshold = 800   # distance covered during invalid transition
        min_velocity_threshold = 500       # minimum speed during the invalid transition
        dist_prev_next_threshold = 400      # maximum permittted distance between previos and next location for the invalid transition stamp
        time_gap_pre_next_threshold = 480 # Time gap(min) between previuos and next time stamp for the invalid record
        sub_folder_name = f'threshold_dist_{displacement_min_threshold}_vel_{min_velocity_threshold}_dist_prev_next_{dist_prev_next_threshold}_time_gap_{time_gap_pre_next_threshold}_min'
    elif step==9:
        displacement_min_threshold = 900   # distance covered during invalid transition
        min_velocity_threshold = 500       # minimum speed during the invalid transition
        dist_prev_next_threshold = 450      # maximum permittted distance between previos and next location for the invalid transition stamp
        time_gap_pre_next_threshold = 540 # Time gap(min) between previuos and next time stamp for the invalid record
        sub_folder_name = f'threshold_dist_{displacement_min_threshold}_vel_{min_velocity_threshold}_dist_prev_next_{dist_prev_next_threshold}_time_gap_{time_gap_pre_next_threshold}_min'
    elif step==10:
        displacement_min_threshold = 1000   # distance covered during invalid transition
        min_velocity_threshold = 500       # minimum speed during the invalid transition
        dist_prev_next_threshold = 500      # maximum permittted distance between previos and next location for the invalid transition stamp
        time_gap_pre_next_threshold = 600 # Time gap(min) between previuos and next time stamp for the invalid record
        sub_folder_name = f'threshold_dist_{displacement_min_threshold}_vel_{min_velocity_threshold}_dist_prev_next_{dist_prev_next_threshold}_time_gap_{time_gap_pre_next_threshold}_min'
    elif step==11:
        displacement_min_threshold = 1000   # distance covered during invalid transition
        min_velocity_threshold = 500       # minimum speed during the invalid transition
        dist_prev_next_threshold = 550      # maximum permittted distance between previos and next location for the invalid transition stamp
        time_gap_pre_next_threshold = 660 # Time gap(min) between previuos and next time stamp for the invalid record
        sub_folder_name = f'threshold_dist_{displacement_min_threshold}_vel_{min_velocity_threshold}_dist_prev_next_{dist_prev_next_threshold}_time_gap_{time_gap_pre_next_threshold}_min'
    elif step==12:
        displacement_min_threshold = 1000   # distance covered during invalid transition
        min_velocity_threshold = 500       # minimum speed during the invalid transition
        dist_prev_next_threshold = 600      # maximum permittted distance between previos and next location for the invalid transition stamp
        time_gap_pre_next_threshold = 720 # Time gap(min) between previuos and next time stamp for the invalid record
        sub_folder_name = f'threshold_dist_{displacement_min_threshold}_vel_{min_velocity_threshold}_dist_prev_next_{dist_prev_next_threshold}_time_gap_{time_gap_pre_next_threshold}_min'
    elif step==13:
        displacement_min_threshold = 1000   # distance covered during invalid transition
        min_velocity_threshold = 500       # minimum speed during the invalid transition
        dist_prev_next_threshold = 650      # maximum permittted distance between previos and next location for the invalid transition stamp
        time_gap_pre_next_threshold = 780 # Time gap(min) between previuos and next time stamp for the invalid record
        sub_folder_name = f'threshold_dist_{displacement_min_threshold}_vel_{min_velocity_threshold}_dist_prev_next_{dist_prev_next_threshold}_time_gap_{time_gap_pre_next_threshold}_min'
    elif step==14:
        displacement_min_threshold = 1000   # distance covered during invalid transition
        min_velocity_threshold = 500       # minimum speed during the invalid transition
        dist_prev_next_threshold = 700      # maximum permittted distance between previos and next location for the invalid transition stamp
        time_gap_pre_next_threshold = 840 # Time gap(min) between previuos and next time stamp for the invalid record
        sub_folder_name = f'threshold_dist_{displacement_min_threshold}_vel_{min_velocity_threshold}_dist_prev_next_{dist_prev_next_threshold}_time_gap_{time_gap_pre_next_threshold}_min'
    if step==15:
        displacement_min_threshold = 50   # distance covered during invalid transition
        min_velocity_threshold = 100       # minimum speed during the invalid transition
        dist_prev_next_threshold = 25      # maximum permittted distance between previos and next location for the invalid transition stamp
        time_gap_pre_next_threshold = 30 # Time gap(min) between previuos and next time stamp for the invalid record
        sub_folder_name = f'threshold_dist_{displacement_min_threshold}_vel_{min_velocity_threshold}_dist_prev_next_{dist_prev_next_threshold}_time_gap_{time_gap_pre_next_threshold}_min'
    if step==16:
        displacement_min_threshold = 50   # distance covered during invalid transition
        min_velocity_threshold = 100       # minimum speed during the invalid transition
        dist_prev_next_threshold = 15      # maximum permittted distance between previos and next location for the invalid transition stamp
        time_gap_pre_next_threshold = 15 # Time gap(min) between previuos and next time stamp for the invalid record
        sub_folder_name = f'threshold_dist_{displacement_min_threshold}_vel_{min_velocity_threshold}_dist_prev_next_{dist_prev_next_threshold}_time_gap_{time_gap_pre_next_threshold}_min'
    else:
        pass
    return displacement_min_threshold, min_velocity_threshold, dist_prev_next_threshold, time_gap_pre_next_threshold, sub_folder_name


def remove_type_1(df, save_path_type_1):
    # for old whole data
    os.makedirs(save_path_type_1, exist_ok=True)
    record={}
    step =1
    while step<17:
        print(f'records before filtering invalid stamps : {df.shape}')
        displacement_min_threshold, min_velocity_threshold, dist_prev_next_threshold, time_gap_pre_next_threshold, sub_folder_name = params(step)
        # data_new.to_excel(fr'{folder_path}\data_{folder_name}.xlsx',index=False)  #save_original_dataframe
    
        # state validity_of_records
        df = calculate_distances_and_time_gaps_continous_invalid_stamps(df, displacement_min_threshold , min_velocity_threshold,time_gap_pre_next_threshold, dist_prev_next_threshold  )  #dataframe, displacement(km), velocity(kmph)
        # rec_with_issue.to_csv(fr'C:\Users\User\Downloads\Mobility_traces_new_data_code\validating_records\filtering issue from records with in-between timestamps\stepwise processed records\Type 1\{step}_{sub_folder_name}.csv',index=False)  #save datframe processed at stage 1

        df = filter_out_continuos_invalid_stamps(df)  # dataframe, distance_between previpus and next(km), time(minutes)
        print(f'data_new_ shape after filtering out invalid records: {df.shape}')

        df = optimized_function_maid_vectorized(df)


        # plot_cumulative_distance(data_new, f'Filtered:')
        save_path = os.path.join(save_path_type_1, f'df_without_type_1_step_{step}.csv')
        df.to_csv(save_path,index=False)  # save dataframe after filtering out individual records
        step+=1
    save_path_json = os.path.join(save_path_type_1, f'stepwise_count_type_1.json')
    with open(save_path_json, 'w') as f:
        json.dump(record, f)
    return df

    
def remove_type_2(df, save_path_type_2):
    # for old whole data
    os.makedirs(save_path_type_2, exist_ok=True)
    record={}
    step =1
    while step<17:
        print(f'records before filtering invalid stamps : {df.shape}')
        displacement_min_threshold, min_velocity_threshold, dist_prev_next_threshold, time_gap_pre_next_threshold, sub_folder_name = params(step)
        # data_new.to_excel(fr'{folder_path}\data_{folder_name}.xlsx',index=False)  #save_original_dataframe
    
        # state validity_of_records
        df = calculate_distances_and_time_gaps_continous_invalid_stamps_looking_backward(df, displacement_min_threshold , min_velocity_threshold,time_gap_pre_next_threshold, dist_prev_next_threshold  )  #dataframe, displacement(km), velocity(kmph)
        # rec_with_issue.to_csv(fr'C:\Users\User\Downloads\Mobility_traces_new_data_code\validating_records\filtering issue from records with in-between timestamps\stepwise processed records\Type 1\{step}_{sub_folder_name}.csv',index=False)  #save datframe processed at stage 1

        df = filter_out_continuos_invalid_stamps(df)  # dataframe, distance_between previpus and next(km), time(minutes)
        print(f'data_new_ shape after filtering out invalid records: {df.shape}')

        df = optimized_function_maid_vectorized(df)


        # plot_cumulative_distance(data_new, f'Filtered:')
        save_path = os.path.join(save_path_type_2, f'df_without_type_1_step_{step}.csv')
        df.to_csv(save_path,index=False)  # save dataframe after filtering out individual records
        step+=1
    save_path_json = os.path.join(save_path_type_2, f'stepwise_count_type_1.json')
    with open(save_path_json, 'w') as f:
        json.dump(record, f)
    return df

# def remove_type_3(df, save_path_type_2):
import geopandas as gpd
# from geopy.distance import geodesic
from datetime import timedelta
from tqdm import tqdm
from shapely.geometry import Point

def find_airports(df):
    airports = gpd.read_file(r"C:\Users\User\Downloads\Mobility_traces_codes\Indian airports\boundary\airports_boundary.shp")
    airports = airports[['osm_id', 'name_en','geometry']]
    df['geometry'] = df.apply(lambda row: Point(row['longitude'], row['latitude']), axis=1)
    gdf= gpd.GeoDataFrame(df, geometry='geometry')
    gdf = gdf.set_crs("EPSG:4326")
    gdf = gdf.to_crs("EPSG:3857")
    gdf = gpd.sjoin(gdf, airports, how="left", predicate='within')
    return gdf



def remove_type_3(df, velocity_threshold=150, displacement_threshold=150, min_velocity=60, max_velocity=90):
    """
    Process the DataFrame to find and flag records based on given conditions, and adjust datetimes if needed.

    Args:
    - df (DataFrame): The input DataFrame with data.
    - velocity_threshold (float): The threshold for velocity to consider a record.
    - displacement_threshold (float): The threshold for displacement to consider a record.
    - min_velocity (float): The minimum velocity for the ratio of distance to time difference.
    - max_velocity (float): The maximum velocity for the ratio of distance to time difference.

    Returns:
    - DataFrame: The processed DataFrame with records flagged for keeping or removal.
    """

    df = find_airports(df)
    # Ensure the datetime column is in datetime format
    df['datetime'] = pd.to_datetime(df['datetime'])
    
    # Add new columns for actions chosen and keep flag
    df['Action chosen'] = 'None'
    df['keep'] = True

    # Calculate geodesic distance
    # def calculate_geodesic_distance(lat1, lon1, lat2, lon2):
    #     return geodesic((lat1, lon1), (lat2, lon2)).kilometers

    # Calculate time difference in hours
    def calculate_time_diff_hours(time1, time2):
        return abs((time1 - time2).total_seconds()) / 3600

    # Function to process each group
    def process_group(group):
        max_distance_transition_while_tracing = 20
        # Reset index for group processing
        group = group.reset_index(drop=True)
        
        # Filter relevant records based on velocity and displacement thresholds
        relevant_indices = group[(group['Velocity'] > velocity_threshold) & (group['displacement'] > displacement_threshold)].index
        count = len(relevant_indices)
        # print(f'Group size: {len(group)}, Relevant records count: {count}')
        
        # Process each relevant record
        for i in range(count):
            index = relevant_indices[i]
            record = group.iloc[index]
            # print(f'\nProcessing record at index {index}: {record.to_dict()}')
            
            # Set initial velocity range
            current_min_velocity = min_velocity
            current_max_velocity = max_velocity
            
            # Check distance from airport
            if index > 0:
                distance_from_airport_i = group.loc[index, 'osm_id']
                distance_from_airport_prev = group.loc[index-1, 'osm_id']
                # print(f'Distance from airport (current): {distance_from_airport_i}, (previous): {distance_from_airport_prev}')
                
                if distance_from_airport_i  and distance_from_airport_prev :
                    current_min_velocity = 500
                    current_max_velocity = 600
                    max_distance_transition_while_tracing = 2
                    # print(f'Updated velocity range: {current_min_velocity} - {current_max_velocity}')
            
            valid_record_found = False
            # Iterate backwards from index i to find a valid previous record
            for l in range(index - 1, -1, -1):
                if index - l > 1:
                    distance_change = euclidean(
                        (group.loc[l, 'x'], group.loc[l, 'y']),
                        (group.loc[l + 1, 'x'], group.loc[l + 1, 'y'])
                    )
                    # print(f'Distance change between index {l} and {l+1}: {distance_change} km')
                    
                    if distance_change < max_distance_transition_while_tracing:
                        distance = euclidean(
                            (group.loc[index, 'x'], group.loc[index, 'y']),
                            (group.loc[l, 'x'], group.loc[l, 'y'])
                        )
                        time_diff_hours = calculate_time_diff_hours(
                            group.loc[index, 'datetime'], group.loc[l, 'datetime']
                        )
                        # print(f'Distance to previous record: {distance} km, Time difference: {time_diff_hours} hours')
                        
                        if time_diff_hours > 0 and distance >= 0:
                            velocity_ratio = distance / time_diff_hours
                            # print(f'Calculated velocity ratio: {velocity_ratio}')
                        elif time_diff_hours == 0 and distance ==0:
                            velocity_ratio = 0
                            # print(f'Calculated velocity ratio: {velocity_ratio}')                            
                        elif time_diff_hours== 0:
                            velocity_ratio = np.inf
                            # 
                        # print(f'Calculated velocity ratio: {velocity_ratio}')     


                        if current_min_velocity < velocity_ratio < current_max_velocity:
                            group.at[l, 'Action chosen'] = 'selected'
                            group.loc[l + 1:index-1, 'keep'] = False
                            valid_record_found = True
                            # print(f'Record at index {l} selected.')
                            break

                        elif velocity_ratio < current_min_velocity:
                            # Calculate time x for comparison
                            distance_l_l1 = euclidean(
                                (group.loc[l+1, 'x'], group.loc[l+1, 'y']),
                                (group.loc[l, 'x'], group.loc[l, 'y'])
                            )
                            time_x = group.loc[l, 'datetime'] + timedelta(hours=distance_l_l1 / min_velocity)
                            # print(f'Time x for record at index {l+1}: {time_x}')

                            # Calculate new time for record l
                            distance_l_plus_1_index = euclidean(
                                (group.loc[l+1, 'x'], group.loc[l+1, 'y']),
                                (group.loc[index, 'x'], group.loc[index, 'y'])
                            )
                            new_time = group.loc[index, 'datetime'] - timedelta(hours=distance_l_plus_1_index / min_velocity)
                            # print(f'New time for record at index {l+1}: {new_time}')

                            if new_time >= time_x:
                                group.at[l+1, 'datetime'] = new_time
                                group.at[l+1, 'Action chosen'] = 'modified'
                                
                                # Mark records from l+2 to index-1 for removal
                                group.loc[l + 2:index-1, 'keep'] = False
                                valid_record_found = True
                                # print(f'Record at index {l+1} modified.')
                                break
                    elif distance_change >= max_distance_transition_while_tracing:
                        # print(f'when distance>=20')
                        distance_l_l1 = euclidean(
                            (group.loc[l+1, 'x'], group.loc[l+1, 'y']),
                            (group.loc[l, 'x'], group.loc[l, 'y'])
                        )
                        time_x = group.loc[l, 'datetime'] + timedelta(hours=distance_l_l1 / min_velocity)
                        # print(f'Time x for record at index {l+1}: {time_x}')

                        distance_l_plus_1_index = euclidean(
                            (group.loc[l+1, 'x'], group.loc[l+1, 'y']),
                            (group.loc[index, 'x'], group.loc[index, 'y'])
                        )
                        new_time = group.loc[index, 'datetime'] - timedelta(hours=distance_l_plus_1_index / min_velocity)
                        # print(f'New time for record at index {l+1}: {new_time}')
                        if new_time >= time_x:
                            group.at[l+1, 'datetime'] = new_time
                            group.at[l+1, 'Action chosen'] = 'modified'
                            
                            # Mark records from l+2 to index-1 for removal
                            group.loc[l + 2:index-1, 'keep'] = False
                            valid_record_found = True
                            # print(f'Record at index {l+1} modified.')
                            break
                        else:
                            group.at[index, 'Action chosen'] = 'Remained same'
                            break


            if not valid_record_found:
                # If no valid modification was made, mark as 'Remained same'
                group.at[index, 'Action chosen'] = 'Remained same'
                # print(f'Record at index {index} remained the same.')

        return group

    grouped = df.groupby('maid')
    # Process each group with a progress bar
    processed_groups = []
    for _, group in tqdm(grouped, total=len(grouped), desc="Processing Groups"):
        processed_groups.append(process_group(group))

    # Combine the processed groups back into a single DataFrame
    df = pd.concat(processed_groups)

    return df