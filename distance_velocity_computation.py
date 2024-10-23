import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
import warnings
from geopy.distance import geodesic
from tqdm import tqdm

warnings.filterwarnings("ignore")

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

                    distance = geodesic(
                        (prev_record['latitude'], prev_record['longitude']),
                        (next_record['latitude'], next_record['longitude'])
                    ).kilometers

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
                        indices_to_mark = range(original_index , original_next_valid_index)
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
# def plot_graph(df, save_path, plot_title, displacement_col='displacement', time_gap_col='consecutive_time_gap(minutes)'):
#     """
#     Plot displacement and time gap graphs from the DataFrame and save them.

#     Parameters:
#     - df (pd.DataFrame): DataFrame with 'displacement' and 'consecutive_time_gap(minutes)' columns.
#     - save_path (str): Directory path to save the plots.
#     - plot_title (str): Title for the plots.
#     - displacement_col (str): Column name for displacement values.
#     - time_gap_col (str): Column name for time gap values.
#     """
#     df['datetime'] = pd.to_datetime(df['datetime'])

#     # Plot Displacement
#     plt.figure(figsize=(10, 6))
#     plt.plot(df['datetime'], df[displacement_col], marker='o', linestyle='-', color='b', label='Displacement')
#     plt.xlabel('Datetime')
#     plt.ylabel('Displacement')
#     plt.title(f'{plot_title} - Displacement')
#     plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M:%S'))
#     plt.gca().xaxis.set_major_locator(mdates.HourLocator(interval=1))
#     plt.xticks(rotation=45)
#     plt.legend()
#     plt.tight_layout()
#     plt.savefig(os.path.join(save_path, f'{plot_title}_displacement.png'))
#     plt.close()

#     # Plot Time Gap
#     plt.figure(figsize=(10, 6))
#     plt.plot(df['datetime'], df[time_gap_col], marker='o', linestyle='-', color='r', label='Time Gap')
#     plt.xlabel('Datetime')
#     plt.ylabel('Time Gap (minutes)')
#     plt.title(f'{plot_title} - Time Gap')
#     plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M:%S'))
#     plt.gca().xaxis.set_major_locator(mdates.HourLocator(interval=1))
#     plt.xticks(rotation=45)
#     plt.legend()
#     plt.tight_layout()
#     plt.savefig(os.path.join(save_path, f'{plot_title}_time_gap.png'))
#     plt.close()
