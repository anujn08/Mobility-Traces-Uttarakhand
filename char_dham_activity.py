import pandas as pd
import geopandas as gpd
import numpy as np
from shapely.geometry import Point
from shapely import Point
from datetime import timedelta
from tqdm import tqdm
import matplotlib.dates as mdates
# import seaborn as sns

from scipy.stats import skew, kurtosis
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

# Data: List of dictionaries containing name, latitude, and longitude
data = [
    {"name": "Kedarnath", "latitude": 30.7346, "longitude": 79.0669},
    {"name": "Tungnath", "latitude": 30.48944, "longitude": 79.215278},
    # {"name": "Rudranath", "latitude": 30.5194, "longitude": 79.31833},
    # {"name": "Madhyamaheshwar", "latitude": 30.63694, "longitude": 79.21611},
    # {"name": "Kalpeshwar", "latitude": 30.57704, "longitude": 79.422913},
    {"name": "Badrinath", "latitude": 30.743309, "longitude": 79.493767},
    {"name": "Gangotri", "latitude": 30.9947, "longitude": 78.9398},
    {"name": "Yamunotri", "latitude": 31.014, "longitude": 78.4600},
    
    {"name": "Gaurikund", "latitude": 30.1557, "longitude": 79.3918},
    
    {"name": "Sonprayag", "latitude": 30.6325, "longitude": 78.9953},
    
    {"name": "Janki Chatti", "latitude": 30.9753, "longitude": 78.4361},
    
    {"name": "Guptkashi", "latitude": 30.5229, "longitude": 79.077}
]

# Convert data to GeoDataFrame
gdf_char_dham = gpd.GeoDataFrame(
    data,
    geometry=[Point(lon, lat) for lon, lat in zip([d["longitude"] for d in data], [d["latitude"] for d in data])],
    crs="EPSG:4326"  # Coordinate Reference System (WGS 84)
)

gdf_char_dham.rename(columns = {'latitude':'latitude_char_dham', 'longitude':'longitude_char_dham'}, inplace=True)

gdf_char_dham = gdf_char_dham.to_crs("EPSG:3857")

def merge_char_dham_important_locations(df, dist_col, lat_column='latitude', lon_column='longitude'):
    """
    Merges a dataframe containing latitude and longitude information with a shapefile.

    Parameters:
    df (pd.DataFrame): Input dataframe containing 'latitude' and 'longitude' columns (or custom columns defined).
    lat_column (str): Name of the latitude column in the dataframe. Defaults to 'latitude'.
    lon_column (str): Name of the longitude column in the dataframe. Defaults to 'longitude'.

    Returns:
    pd.DataFrame: Dataframe merged with the shapefile attributes based on spatial join.
    """

    # Step 2: Convert the latitude and longitude columns in the dataframe to a GeoDataFrame
    # Create 'geometry' column by combining the latitude and longitude values into Point objects
    geometry = [Point(xy) for xy in zip(df[lon_column], df[lat_column])]
    
    # Create a GeoDataFrame from the original dataframe
    gdf = gpd.GeoDataFrame(df.copy(), geometry=geometry, crs="EPSG:4326")  # Assuming WGS84 Coordinate Reference System (EPSG:4326)

    # Ensure the CRS matches
    gdf.set_crs(epsg=4326, inplace=True)
    gdf = gdf.to_crs("EPSG:3857")
    

    # Step 3: Perform a spatial join between the GeoDataFrame and the shapefile GeoDataFrame
    # Using 'inner' join to keep only points that intersect with the polygons in the shapefile
    gdf = gpd.sjoin_nearest(gdf, gdf_char_dham, how="left", distance_col=dist_col)
    gdf.reset_index(drop=True)

    gdf = gdf.sort_values(by=['maid', 'datetime',lat_column, lon_column, dist_col])
    gdf = gdf.drop_duplicates(subset=['maid', 'datetime', lat_column, lon_column], keep='first')

    # return gdf

    # Step 4: Drop the geometry columns if you want a regular DataFrame after merging
    gdf = pd.DataFrame(gdf.drop(columns=['geometry', 'index_right', 'latitude_char_dham', 'longitude_char_dham' ]))

        
    # Step 1: Initialize the new column 'Chaar_dham_location' with default values (optional)
    gdf['Chaar_dham_location'] = None  # Or another default value if desired

    # Step 2: Update 'Chaar_dham_location' based on the condition
    gdf.loc[gdf[dist_col] < 6000, 'Chaar_dham_location'] = gdf['name']


    return gdf

tqdm.pandas()

# Function to calculate 'time_spent' for each group
def calculate_time_spent(group):
    # Initialize variables to track the current group, start time, and start index
    current_group = None
    start_time_group = None
    start_index_group = None
    
    # Initialize variables for district
    current_district = None
    start_time_district = None
    start_index_district = None
    
    # Initialize variables for Chaar Dham activity
    current_chaar_dham_activity = None
    start_time_chaar_dham = None
    start_index_chaar_dham = None
    
    # Iterate over each row in the group
    for i in range(len(group) - 1):
        # Position group variables
        group_i = group.iloc[i]['position_group']
        group_next = group.iloc[i + 1]['position_group']
        
        # District variables
        district_i = group.iloc[i]['district']
        district_next = group.iloc[i + 1]['district']
        
        # Chaar Dham activity variables
        activity_i = group.iloc[i]['Chaar_dham_location']
        activity_next = group.iloc[i + 1]['Chaar_dham_location']
        
        # Check for position group change
        if group_i != current_group:
            if current_group is not None and start_time_group is not None:
                # Calculate the feasible time to reach the next point based on the previous logic
                x_i, y_i = group.iloc[i]['x'], group.iloc[i]['y']
                x_prev, y_prev = group.iloc[i - 1]['x'], group.iloc[i - 1]['y']

                # Calculate the Euclidean distance between the two points in meters
                distance_m = np.sqrt((x_prev - x_i) ** 2 + (y_prev - y_i) ** 2)

                # Convert distance from meters to kilometers
                distance_km = distance_m / 1000

                # Calculate time gap that can be taken to reach the next point
                time_gap = timedelta(hours=distance_km / 75)

                # Calculate the feasible time to reach the next point
                feasible_time = group.iloc[i]['datetime'] - time_gap

                # Calculate the difference between start_time and feasible_time
                if feasible_time > start_time_group:
                    time_difference = feasible_time - start_time_group
                    group.at[start_index_group, 'time_spent_in_position_group'] = time_difference

            # Update current group tracking variables
            current_group = group_i
            start_time_group = group.iloc[i]['datetime']
            start_index_group = group.index[i]

        # Check for district change
        if district_i != current_district:
            if current_district is not None and start_time_district is not None:
                x_i, y_i = group.iloc[i]['x'], group.iloc[i]['y']
                x_prev, y_prev = group.iloc[i - 1]['x'], group.iloc[i - 1]['y']

                # Calculate the Euclidean distance between the two points in meters
                distance_m = np.sqrt((x_prev - x_i) ** 2 + (y_prev - y_i) ** 2)

                # Convert distance from meters to kilometers
                distance_km = distance_m / 1000

                # Calculate time gap that can be taken to reach the next point
                time_gap = timedelta(hours=distance_km / 75)

                # Calculate the feasible time to reach the next point
                feasible_time = group.iloc[i]['datetime'] - time_gap
                
                # Calculate the difference between start_time and feasible_time
                if feasible_time > start_time_district:
                    time_difference = feasible_time - start_time_district
                    group.at[start_index_district, 'time_spent_in_district'] = time_difference

            # Update current district tracking variables
            current_district = district_i
            start_time_district = group.iloc[i]['datetime']
            start_index_district = group.index[i]
        
        # Check for Chaar Dham activity change
        if activity_i != current_chaar_dham_activity:
            if current_chaar_dham_activity is not None and start_time_chaar_dham is not None:
                # Calculate time spent in the current activity
                x_i, y_i = group.iloc[i]['x'], group.iloc[i]['y']
                x_prev, y_prev = group.iloc[i - 1]['x'], group.iloc[i - 1]['y']

                # Calculate the Euclidean distance between the two points in meters
                distance_m = np.sqrt((x_prev - x_i) ** 2 + (y_prev - y_i) ** 2)

                # Convert distance from meters to kilometers
                distance_km = distance_m / 1000

                # Calculate time gap that can be taken to reach the next point
                time_gap = timedelta(hours=distance_km / 75)

                # Calculate the feasible time to reach the next point
                feasible_time = group.iloc[i]['datetime'] - time_gap
                
                # Calculate the difference between start_time and feasible_time
                if feasible_time > start_time_chaar_dham:
                    time_difference = feasible_time - start_time_chaar_dham
                    group.at[start_index_chaar_dham, 'time_spent_in_chaar_dham_activity'] = time_difference

            # Update current Chaar Dham activity tracking variables
            current_chaar_dham_activity = activity_i
            start_time_chaar_dham = group.iloc[i]['datetime']
            start_index_chaar_dham = group.index[i]
            
    # Handle the last segment where no change occurred
    if start_time_group is not None:
        time_difference = group.iloc[-1]['datetime'] - start_time_group
        group.at[start_index_group, 'time_spent_in_position_group'] = time_difference
        
    if start_time_district is not None:
        time_difference = group.iloc[-1]['datetime'] - start_time_district
        group.at[start_index_district, 'time_spent_in_district'] = time_difference
        
    if start_time_chaar_dham is not None:
        time_difference = group.iloc[-1]['datetime'] - start_time_chaar_dham
        group.at[start_index_chaar_dham, 'time_spent_in_chaar_dham_activity'] = time_difference

    return group

def char_dham_visitors(df):
    # print(df.columns)
    df['datetime'] = pd.to_datetime(df['datetime'])
    try:

        # Check if the required columns exist in the DataFrame
        if 'Chaar_dham_location' not in df.columns or 'maid' not in df.columns:
            raise ValueError("DataFrame must contain 'Chaar_dham_location' and 'maid' columns.")
        # print(f'length : {len(df)}')
        chardham_visitors = df[df['Chaar_dham_location'].notna()]['maid'].unique().tolist()
        # print(f'chardham_visitors: {chardham_visitors}')
        
        # Create a DataFrame for chardham visitors
        df_chardham_visitors = df[df['maid'].isin(set(chardham_visitors))]
        
        # Group by 'maid' and apply the time calculation
        df_chardham_visitors = df_chardham_visitors.groupby('maid').progress_apply(calculate_time_spent)
        
        return df_chardham_visitors

    except Exception as e:
        print(f"An error occurred: {e}")
        return None  # or you can return an empty DataFrame or handle it as needed


# Step 2: Count IDs per path
from collections import Counter

def number_of_different_paths_taken(df):
    # Step 1: Find paths per ID
    paths = {}
    for device_id, group in df.groupby('maid'):
        # Get the district path by dropping consecutive duplicates
        path = group['district'].loc[group['district'].shift() != group['district']].tolist()
        paths[device_id] = path



    # Convert paths dictionary values to tuples (immutable, hashable)
    path_counts = Counter(tuple(v) for v in paths.values())

    # Sort the path_counts items by count in descending order
    sorted_paths = sorted(path_counts.items(), key=lambda item: item[1], reverse=True)   
    total_IDs = df['maid'].nunique()
    print(f'total paths: {len(path_counts)}')
    print(f'Total IDs: {total_IDs}')
    df_sorted_paths = pd.DataFrame(sorted_paths, columns=['Path', 'Count'])
    df_sorted_paths = df_sorted_paths[['Count', 'Path']]
    return df_sorted_paths


def plot_travel_duration(df, district_col, time_spent_col, title, ax):
    # Ensure 'datetime' column is in datetime format
    df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')

    # Convert 'time_spent_in_district' to timedelta if not already
    df[time_spent_col] = pd.to_timedelta(df[time_spent_col], errors='coerce')

    # Remove rows where 'time_spent_in_district' or 'datetime' is NaT (Not a Time)
    filtered_df = df.dropna(subset=[time_spent_col, 'datetime'])

    # Check if filtered_df is empty
    if filtered_df.empty:
        ax.set_title(f'{title} (No Data)')
        return

    # Create a dictionary to store start_time and end_time per district
    district_times = {}

    # Populate the dictionary
    for _, row in filtered_df.iterrows():
        district = row[district_col]
        start_time = row['datetime']
        time_spent = row[time_spent_col]
        end_time = start_time + time_spent

        if district not in district_times:
            district_times[district] = []
        
        district_times[district].append((start_time, end_time))

    # Define bar height
    bar_height = 0.25

    # Plot using the dictionary
    for district, times in district_times.items():
        for start_time, end_time in times:
            # Ensure that start_time and end_time are valid and within range
            if not pd.isna(start_time) and not pd.isna(end_time) and start_time < end_time:
                ax.barh(district, end_time - start_time,
                        left=start_time, height=bar_height, 
                        align='center', color=plt.cm.tab10(hash(district) % 10))

    # Set x-axis limits to start from the minimum date and cover the data range
    start_date = df['datetime'].min()
    end_date = df['datetime'].max()

    if pd.isna(start_date) or pd.isna(end_date):
        ax.set_title(f'{title} (Invalid Date)')
        return

    ax.set_xlim([start_date, end_date])

    # Set the x-axis major locator to day intervals and format date labels
    ax.xaxis.set_major_locator(mdates.DayLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d'))

    # Set x-axis minor locator to 1-hour intervals for grid lines
    ax.xaxis.set_minor_locator(mdates.HourLocator(interval=24))

    # Add grid lines for both major and minor ticks
    ax.grid(which='both', axis='x', linestyle='--', color='gray', alpha=0.7)

    # Rotate x-axis labels for better readability
    ax.tick_params(axis='x', labelrotation=45)

    # Add labels and title
    ax.set_xlabel('Time')
    ax.set_ylabel(district_col)
    ax.set_title(title)


def plot_for_multiple_ids(df, fixed_maids, plot_save_path):
    # Fixed list of three maids
    # fixed_maids = ['019d1949-0979-4fe0-ba16-3544e2caf822', '02ae48b0-af10-403b-aecd-6a70d92efd58', '0327649e-add6-4c86-8399-6604d75b1509']

    # Create 3x3 subplots
    fig, axs = plt.subplots(3, 2, figsize=(20, 15))

    # Flatten the 2D array of axs to 1D for easy iteration
    axs = axs.flatten()

    # Titles for each plot
    titles = [
        'Time spent in each district',
        'Time spent at each char dham'
    ]
    cols = ['time_spent_in_district', 'time_spent_in_chaar_dham_activity']

    # Plot for each maid and each category
    for i, maid in enumerate(fixed_maids):
        for j, (col, title) in enumerate(zip(['district', 'Chaar_dham_location'], titles)):
            ax_index = i * 2 + j
            if ax_index < len(axs):  # Ensure we don't access out-of-bounds axs
                ax = axs[ax_index]
                # Filter out unwanted '000' entries for specific columns
                if col == 'Chaar_dham_location':
                    df_filtered = df[(df['maid'] == maid) & (df['Chaar_dham_location'].notna())]
                else:
                    df_filtered = df[df['maid'] == maid]
                    
                plot_travel_duration(
                    df_filtered,
                    district_col=col,
                    time_spent_col=cols[j],
                    title=f'Maid {maid} - {title}',
                    ax=ax
                )

    # Hide unused subplots if any
    for k in range(len(fixed_maids) * 3, len(axs)):
        axs[k].axis('off')

    # Adjust layout
    plt.tight_layout()

        # Save the plot to the specified path
    plt.savefig(plot_save_path, bbox_inches='tight', dpi=300)


import pandas as pd
from itertools import chain, combinations

def get_activity_subset_counts(df, activity_column='Chaar_dham_location', group_column='maid'):
    """
    Generates a count of unique devices (maids) for each subset of activities and returns as a DataFrame.

    Parameters:
    df (pd.DataFrame): DataFrame containing the activity and grouping information.
    activity_column (str): Column name representing activities (default is 'Chaar_dham_location').
    group_column (str): Column name representing device/grouping ID (default is 'maid').

    Returns:
    pd.DataFrame: DataFrame with 'Activity Subset' and 'Device Count' columns.
    """
    # Step 1: Identify unique activities
    unique_activities = df[activity_column].unique().tolist()

    # Step 2: Generate all subsets of unique activities
    all_subsets = list(chain.from_iterable(combinations(unique_activities, r) for r in range(1, len(unique_activities) + 1)))

    # Step 3: Group by device/group and get unique activities per device
    activity_per_device = df.groupby(group_column)[activity_column].apply(lambda x: tuple(set(x)))

    # Step 4: Count devices for each activity subset
    subset_counts = {
        subset: activity_per_device[activity_per_device.apply(lambda x: set(x) == set(subset))].nunique()
        for subset in all_subsets
    }

    # Step 5: Convert to DataFrame for easy visualization
    subset_counts_df = pd.DataFrame(list(subset_counts.items()), columns=['Activity Subset', 'Device Count'])

    return subset_counts_df

import pandas as pd

def calculate_origin_state_counts(df, subset_counts_df, activity_col='Chaar_dham_location', origin_state_col='origin_state', maid_col='maid'):
    """
    Calculates the count of unique devices (maids) from each origin state for various activity subsets.

    Parameters:
    df (pd.DataFrame): The original DataFrame containing the data.
    subset_counts_df (pd.DataFrame): DataFrame containing the subsets of activities and their counts.
    activity_col (str): Name of the column that holds the activity information. Default is 'Chaar_dham_location'.
    origin_state_col (str): Name of the column that holds the origin state information. Default is 'origin_state'.
    maid_col (str): Name of the column that holds the unique device ID information. Default is 'maid'.

    Returns:
    pd.DataFrame: A DataFrame where each row corresponds to an activity subset and columns represent counts of devices from each origin state.
    """
    # Prepare an empty DataFrame to store results
    results_df = pd.DataFrame()

    # Iterate over each subset in the subset_counts DataFrame
    for subset in subset_counts_df['Activity Subset']:
        # Convert subset to set for comparison
        subset_set = set(subset)
        
        # Group by 'maid' and aggregate the unique 'activity_col'
        activity_per_maid = df.groupby(maid_col)[activity_col].unique()
        
        # Filter IDs whose distinct activities match exactly the subset
        matching_ids = activity_per_maid[activity_per_maid.apply(lambda x: set(x) == subset_set)].index
        
        # Filter the original dataframe for the matching IDs
        filtered_df = df[df[maid_col].isin(matching_ids)]
        
        # Group by 'origin_state' and count the unique IDs
        count_per_state = filtered_df.groupby(origin_state_col)[maid_col].nunique()
        
        # Convert to a dictionary to ensure all states are included
        all_states = df[origin_state_col].unique()
        count_per_state_dict = {state: count_per_state.get(state, 0) for state in all_states}
        
        # Convert the dictionary to a DataFrame
        subset_df = pd.DataFrame.from_dict(count_per_state_dict, orient='index', columns=[str(subset)])
        
        # Append to the results dataframe
        results_df = pd.concat([results_df, subset_df], axis=1)

    # Reset index and rename it to 'Origin State'
    results_df.reset_index(inplace=True)
    results_df.rename(columns={'index': 'Origin State'}, inplace=True)
    
    # Transpose the DataFrame to interchange columns with rows
    results_df = results_df.set_index('Origin State').transpose().reset_index()

    return results_df



def plot_heatmap(df, save_path, title='Heatmap of Origin State to Char Dham Count', cmap='viridis', figsize=(12, 8)):
    """
    Creates a heatmap of the given DataFrame, sorts it by rows and columns, and saves the plot.

    Parameters:
    - df (pd.DataFrame): The DataFrame to plot as a heatmap.
    - save_path (str): Path where the plot will be saved.
    - title (str): Title of the heatmap. Default is 'Heatmap of Origin State to Char Dham Count'.
    - cmap (str): Color map to be used for the heatmap. Default is 'viridis'.
    - figsize (tuple): Size of the figure. Default is (12, 8).

    Returns:
    - None
    """
    # Sort rows by total sum in descending order
    df_sorted_rows = df.loc[df.sum(axis=1).sort_values(ascending=False).index]

    # Sort columns by total sum in descending order
    df_sorted = df_sorted_rows.loc[:, df_sorted_rows.sum().sort_values(ascending=False).index]

    # Create a figure and heatmap
    plt.figure(figsize=figsize)
    sns.heatmap(df_sorted, annot=True, cmap=cmap, linewidths=0.5)

    # Set title and show plot
    plt.title(title)

    # Save the plot
    plt.savefig(save_path)

    # Display the plot
    plt.show()


import pandas as pd

def create_od_matrix(df, maid_col='maid', district_col='district', activity_col='Chaar_dham_location'):
    """
    Creates an Origin-Destination (OD) matrix by grouping the DataFrame based on the first district and Chaar Dham activity.

    Parameters:
    - df (pd.DataFrame): The input DataFrame containing 'maid', 'district', and 'Chaar_dham_location' columns.
    - maid_col (str): The column name representing unique IDs (default is 'maid').
    - district_col (str): The column name representing districts (default is 'district').
    - activity_col (str): The column name representing activities (default is 'Chaar_dham_location').

    Returns:
    - pd.DataFrame: A pivot table representing the OD matrix.
    """
    # Step 1: Add a column indicating the first district visited by each unique ID
    df['first_district'] = df.groupby(maid_col)[district_col].transform('first')

    # Step 2: Group by 'first_district' and 'Chaar_dham_location' and count unique 'maid'
    visitor_counts = df.groupby(['first_district', activity_col])[maid_col].nunique().reset_index()

    # Step 3: Create an OD matrix using pivot_table
    od_matrix = visitor_counts.pivot_table(index='first_district', 
                                           columns=activity_col, 
                                           values=maid_col, 
                                           fill_value=0)

    # Step 4: Reset index to make 'first_district' a column again (optional)
    od_matrix = od_matrix.reset_index()

    return od_matrix

from collections import defaultdict
import pandas as pd

def origin_state_origin_district_to_char_dham(df):
    """
    Creates a DataFrame that maps first districts and origin states to unique Chaar dham activities
    with associated maid IDs from the provided DataFrame.

    Parameters:
    df (pd.DataFrame): DataFrame containing columns 'maid', 'district', 
                       'Chaar_dham_location', 'first_district', 
                       'origin_state', and 'datetime'.

    Returns:
    pd.DataFrame: A DataFrame with first districts/states as rows and unique maid IDs as lists for each 
                   Chaar dham activity.
    """
    
    dist_to_char_dham = defaultdict(lambda: defaultdict(list))

    districts = df['first_district'].unique().tolist()
    char_dhams = df['Chaar_dham_location'].unique().tolist()
    states = df['origin_state'].unique().tolist()

    # Find unique records with the earliest datetime for each combination of maid, district, and Chaar_dham_location
    unique_records = df.groupby(['maid', 'district', 'Chaar_dham_location', 'first_district', 'origin_state'])['datetime'].min().reset_index()

    # Initialize the nested dictionary structure with empty lists
    for dist in districts:
        for act in char_dhams:
            if act != '000':
                dist_to_char_dham[dist][act] = []
    for state in states:
        for act in char_dhams:
            if act != '000':
                dist_to_char_dham[state][act] = []

    # Update lists based on unique records
    for index, row in unique_records.iterrows():
        activity = row['Chaar_dham_location']
        first_district = row['first_district']
        first_state = row['origin_state']
        maid = row['maid']
        if activity != '000':
            if first_state == 'UK':
                dist_to_char_dham[first_district][activity].append(maid)
            else:
                dist_to_char_dham[first_state][activity].append(maid)

    # Remove duplicates from lists and sort them
    for dist, activities in dist_to_char_dham.items():
        for act in activities:
            dist_to_char_dham[dist][act] = sorted(set(activities[act]))

    # Convert defaultdict to a regular dict for DataFrame creation
    dist_to_char_dham_dict = {dist: dict(activities) for dist, activities in dist_to_char_dham.items()}

    # Convert the dictionary to a DataFrame
    dist_to_char_dham_df_ids = pd.DataFrame.from_dict(dist_to_char_dham_dict, orient='index')

    # Reset index to make 'first_district' a column again
    dist_to_char_dham_df_ids = dist_to_char_dham_df_ids.reset_index().rename(columns={'index': 'first_district'})

    return dist_to_char_dham_df_ids

def find_travellers_source_to_dest(df, char_dham, district):
    # Step 1: Filter by Chaar_dham_location 'Yamunotri'
    maids_char_dham = df[df['Chaar_dham_location'].isin([char_dham])]['maid'].unique().tolist()
    
    # Filter the DataFrame to only include the maids who visited the specified Chaar_dham_location
    df_filtered = df[df['maid'].isin(maids_char_dham)]

    # Initialize a list to store the IDs fulfilling the condition
    maids_with_condition = []

    # Step 2: Group by each maid to perform ID-wise comparison
    for maid, group in df_filtered.groupby('maid'):
        # Filter the group for the specific char_dham location
        dest_district_records = group[group['Chaar_dham_location'] == char_dham]
        
        # Ensure there is at least one record for the char_dham location
        if not dest_district_records.empty:
            # Get the first record for char_dham
            dest_district_record = dest_district_records.iloc[0]

            # Check if there's any record for the specified district
            district_records = group[group['district'] == district]

            if not district_records.empty:
                # Step 3: Check if the datetime of dest_district is greater than the datetime of the specified district
                if (district_records['datetime'] < dest_district_record['datetime']).any():
                    maids_with_condition.append(maid)

    return maids_with_condition
def find_return_travellers(df, char_dham, district):
    # Step 1: Filter by Chaar_dham_location
    maids_char_dham = df[df['Chaar_dham_location'].isin([char_dham])]['maid'].unique().tolist()

    # Filter the DataFrame to only include the maids who visited the specified Chaar_dham_location
    df_filtered = df[df['maid'].isin(maids_char_dham)]

    # Initialize a list to store the IDs fulfilling the condition
    maids_with_condition = []

    # Step 2: Group by each maid to perform ID-wise comparison
    for maid, group in df_filtered.groupby('maid'):
        # Filter the group for the specific char_dham location
        dest_district_records = group[group['Chaar_dham_location'] == char_dham]
        
        # Ensure there is at least one record for the char_dham location
        if not dest_district_records.empty:
            # Get the first record for char_dham
            dest_district_record = dest_district_records.iloc[0]

            # Check if there's any record for the specified district
            district_records = group[group['district'] == district]

            if not district_records.empty:
                # Step 3: Check if the datetime of dest_district is less than the datetime of the specified district
                if (district_records['datetime'] > dest_district_record['datetime']).any():
                    maids_with_condition.append(maid)

    return maids_with_condition

# Usage example:
# maids_result = find_required_maids(df_chardham_visitors_1, 'Yamunotri', 'HRD')
# print(maids_result)


def find_required_maids_for_lists(df, char_dham, district):
    # Step 1: Filter by Chaar_dham_location 'Yamunotri'
    maids_char_dham = df[df['Chaar_dham_location'].isin(char_dham)]['maid'].unique().tolist()
    df_filtered = df[df['maid'].isin(set(maids_char_dham))]
    # print(df_filtered['district'].unique())
    dest_district = df_filtered[(df_filtered['district']=='RPG') | (df_filtered['district']=='CHM')].iloc[0]
    # dest_district = 'RPG'

    # Step 3: Find maids with the specific 'district'
    maids_with_specific_district = df_filtered[df_filtered['district'] == district]

    # Step 4: Find IDs where datetime of dest_district is greater than datetime of the specific district
    maids_with_condition = maids_with_specific_district[
    maids_with_specific_district['datetime'] > dest_district['datetime']
    ]['maid'].unique().tolist()
    # print(maids_with_specific_district[
    # maids_with_specific_district['datetime'] > dest_district['datetime']
    # ])
    # maids = df_filtered[df_filtered['district'].isin([district])]['maid'].unique().tolist()

    
    return maids_with_condition


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_time_spent_distribution(grouped_df, district, char_dham, reverse_journey=False, save_path=None):
    """
    Plot a histogram for the distribution of time spent in each district with raw frequencies and save the plot as an image.
    
    Parameters:
    grouped_df (pd.DataFrame): DataFrame containing summed time spent in districts (outliers already removed).
    district (str): The source district.
    char_dham (str): The destination Char Dham activity.
    save_path (str, optional): The file path to save the plot image. If None, the plot will not be saved.
    """
    # Find the maximum time spent in any district
    max_time_spent = grouped_df['time_spent_in_district_hours'].max()
    
    # Define bin size and calculate appropriate bin edges
    bin_size = 3
    last_bin_edge = np.ceil(max_time_spent / bin_size) * bin_size
    bins = list(range(0, int(last_bin_edge) + bin_size, bin_size))  # Bins from 0 to last_bin_edge with a gap of bin_size
    bin_labels = [f'{i}h' for i in bins[:-1]] + [f'{int(last_bin_edge)}h+']

    # Get unique districts
    districts = grouped_df['district'].unique()
    
    # Create a subplot for each district
    num_districts = len(districts)
    num_cols = 2  # Number of columns in subplot grid
    num_rows = (num_districts + num_cols - 1) // num_cols  # Compute the number of rows needed

    # Adjust figure size: decrease height (e.g., 4) and increase width (e.g., 16)
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(16, 4 * num_rows), sharex=True, sharey=True)
    
    if reverse_journey:
        # Set the overall title for the plot
        fig.suptitle(f"Time spent distribution for travellers from '{char_dham}' to '{district}'", fontsize=16, y=1.02)
    else:
        fig.suptitle(f"Time spent distribution for travellers from '{district}' to '{char_dham}'", fontsize=16, y=1.02)
    
    # Flatten axes array for easy indexing
    axes = axes.flatten()

    for i, district in enumerate(districts):
        ax = axes[i]
        district_data = grouped_df[grouped_df['district'] == district]
        
        # Create histogram for the district
        hist, bin_edges = np.histogram(district_data['time_spent_in_district_hours'], bins=bins)
        bin_widths = np.diff(bin_edges)
        ax.bar(bin_edges[:-1], hist, width=bin_widths, edgecolor='black', align='edge')
        
        # Set title, labels, and grid
        ax.set_title(f'Distribution of Time Spent in {district}')
        ax.set_xlabel('Time Spent in District (hours)')
        ax.set_ylabel('Frequency')
        ax.grid(True)

        # Set x-ticks to 6-hour intervals but keep bin size of 3 hours
        ax.set_xticks(range(0, int(last_bin_edge) + bin_size * 3, bin_size * 3))
        ax.set_xticklabels([f'{i}h' for i in range(0, int(last_bin_edge) + bin_size * 3, bin_size * 3)], rotation=60)

    # Hide any unused subplots
    for j in range(num_districts, len(axes)):
        axes[j].axis('off')

    # Adjust layout to fit labels
    plt.tight_layout()

    # Save the plot if a save path is provided
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')

    # Show the plot
    plt.show()


import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis

def find_stats_source_to_dest_with_IQR(df, char_dham, district, maids, reverse_journey=False, savepath=None):
    """
    Find statistics for time spent in districts for specified maids between two locations,
    with IQR calculated separately for each district.
    
    Parameters:
    df (pd.DataFrame): The DataFrame containing visit records.
    char_dham (str): The Char Dham activity to filter by.
    district (str): The district to filter by.
    maids (list): List of maid IDs to consider.
    
    Returns:
    tuple: A tuple containing:
        - maids_with_condition (list): List of maids who meet the condition.
        - stats (pd.DataFrame): DataFrame containing statistics for each district.
    """
    # Filter the DataFrame to only include the maids who visited the specified Char Dham activity
    df_filtered = df[df['maid'].isin(maids)]

    # Initialize a list to store the IDs fulfilling the condition
    maids_with_condition = []
    
    # List to store filtered DataFrames
    required_dfs = []
    
    # Group by each maid to perform ID-wise comparison
    for maid, group in df_filtered.groupby('maid'):
        # Find the first datetime record for the specified char_dham
        if char_dham in group['Chaar_dham_location'].values:
            dest_district_record = group[group['Chaar_dham_location'] == char_dham].iloc[0]
            source_district = group[group['district'] == district].iloc[0]
            if reverse_journey:
                # Filter records based on datetime range
                valid_records = group[(group['datetime'] >= dest_district_record['datetime']) & 
                                      (group['datetime'] <= source_district['datetime'])]
            else:
                valid_records = group[(group['datetime'] <= dest_district_record['datetime']) & 
                                      (group['datetime'] >= source_district['datetime'])]

            required_dfs.append(valid_records)
            maids_with_condition.append(maid)
    
    if required_dfs:
        required_df = pd.concat(required_dfs)

        # Ensure 'time_spent_in_district' is a timedelta column
        required_df['time_spent_in_district'] = pd.to_timedelta(required_df['time_spent_in_district'])
        required_df['time_spent_in_district_hours'] = required_df['time_spent_in_district'].dt.total_seconds() / 3600.0

        # Group by 'maid' and 'district', and sum the time spent
        grouped_df = required_df.groupby(['maid', 'district'])['time_spent_in_district_hours'].sum().reset_index()

        # Calculate and print statistics before removing outliers
        district_stats_before = grouped_df.groupby('district')['time_spent_in_district_hours']
        stats_before = district_stats_before.agg(['mean', 'median', 'std', 'count', 'max', 'min']).reset_index()
        stats_before.columns = ['district', 'mean_time_spent', 'median_time_spent', 'std_dev', 'people_count', 'max_time_spent', 'min_time_spent']

        filtered_dfs = []
        outlier_counts = []

        # Calculate IQR and filter data for each district
        for district_name, district_data in grouped_df.groupby('district'):
            # Calculate IQR for the current district
            Q1 = district_data['time_spent_in_district_hours'].quantile(0.25)
            Q3 = district_data['time_spent_in_district_hours'].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Identify outliers
            is_outlier = (district_data['time_spent_in_district_hours'] < lower_bound) | \
                         (district_data['time_spent_in_district_hours'] > upper_bound)
            outlier_count = is_outlier.sum()
            outlier_counts.append(outlier_count)
            
            # Filter data for the current district based on IQR
            filtered_district_data = district_data[~is_outlier]
            filtered_dfs.append(filtered_district_data)

        # Combine filtered data from all districts
        filtered_grouped_df = pd.concat(filtered_dfs)

        # Calculate and print statistics after removing outliers
        district_stats_after = filtered_grouped_df.groupby('district')['time_spent_in_district_hours']
        stats_after = district_stats_after.agg(['mean', 'median', 'std', 'count', 'max', 'min']).reset_index()
        stats_after.columns = ['district', 'mean_time_spent', 'median_time_spent', 'std_dev', 'people_count', 'max_time_spent', 'min_time_spent']

        # Add IQR statistics to the stats_after DataFrame
        stats_after['IQR'] = filtered_grouped_df.groupby('district')['time_spent_in_district_hours'].apply(lambda x: x.quantile(0.75) - x.quantile(0.25)).reset_index(drop=True)
        stats_after['Q1'] = filtered_grouped_df.groupby('district')['time_spent_in_district_hours'].quantile(0.25).reset_index(drop=True)
        stats_after['Q3'] = filtered_grouped_df.groupby('district')['time_spent_in_district_hours'].quantile(0.75).reset_index(drop=True)
        stats_after['Lower_Bound'] = stats_after['Q1'] - 1.5 * stats_after['IQR']
        stats_after['Upper_Bound'] = stats_after['Q3'] + 1.5 * stats_after['IQR']

        # Add outlier counts to the stats_after DataFrame
        stats_after['outlier_count'] = outlier_counts

        # Plot distribution after removing outliers
        # plot_time_spent_distribution(filtered_grouped_df, district, char_dham, reverse_journey, savepath)
    else:
        print("No valid records found for the given criteria.")
        stats_before = pd.DataFrame()
        stats_after = pd.DataFrame()
    
    plot_time_spent_distribution(filtered_grouped_df, district, char_dham, reverse_journey=reverse_journey, save_path=savepath)

    return stats_before, stats_after

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_time_spent_in_chardham(grouped_df, char_dham, save_path=None, state=None):
    """
    Plot a histogram for the distribution of time spent in the specified Chaar Dham activity and save the plot as an image.

    Parameters:
    grouped_df (pd.DataFrame): DataFrame containing summed time spent at Chaar Dham activities (outliers already removed).
    char_dham (str): The destination Chaar Dham activity.
    save_path (str, optional): The file path to save the plot image. If None, the plot will not be saved.
    """
    try:
        # Find the maximum time spent at the Chaar Dham activity
        max_time_spent = grouped_df['time_spent_in_char_dham_Activity_hours'].max()
        
        # Define bin size and calculate appropriate bin edges
        bin_size = 3
        last_bin_edge = np.ceil(max_time_spent / bin_size) * bin_size
        bins = list(range(0, int(last_bin_edge) + bin_size, bin_size))  # Bins from 0 to last_bin_edge with a gap of bin_size
        bin_labels = [f'{i}h' for i in bins[:-1]] + [f'{int(last_bin_edge)}h+']

        # Adjust figure size
        plt.figure(figsize=(10, 6))
        plt.hist(grouped_df['time_spent_in_char_dham_Activity_hours'], bins=bins, edgecolor='black', alpha=0.7)
        if state:
            plt.title(f"{state} :Time Spent Distribution at '{char_dham}'", fontsize=16)
        else:
            plt.title(f"Time Spent Distribution at '{char_dham}'", fontsize=16)
        plt.xlabel('Time Spent (hours)')
        plt.ylabel('Frequency')
        plt.xticks(bins, bin_labels, rotation=45)
        plt.grid(True)
        plt.tight_layout()

        # Save the plot if a save path is provided
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')

        # Show the plot
        # plt.show()

    except Exception as e:
        print(f"Error occurred in plot_time_spent_in_chardham with char_dham='{char_dham}': {e}")


def find_stats_time_spent_in_chardham(df, char_dham, maids, state=None,  savepath=None):
    """
    Find statistics for time spent at Chaar Dham activities for specified maids.

    Parameters:
    df (pd.DataFrame): The DataFrame containing visit records.
    char_dham (str): The Chaar Dham activity to filter by.
    maids (list): List of maid IDs to consider.
    reverse_journey (bool): Whether to consider reverse journeys.
    savepath (str, optional): The file path to save the plot image. If None, the plot will not be saved.

    Returns:
    tuple: A tuple containing:
        - stats_before (pd.DataFrame): DataFrame containing statistics before removing outliers.
        - stats_after (pd.DataFrame): DataFrame containing statistics after removing outliers.
    """
    # Filter the DataFrame to only include the maids who visited the specified Chaar Dham activity
    df_filtered = df[df['maid'].isin(maids)]

    # Extract records for the specified Chaar Dham activity
    required_df = df_filtered[df_filtered['Chaar_dham_location'] == char_dham].copy()

    # Ensure 'time_spent_in_char_dham_activity' is a timedelta column
    required_df['time_spent_in_chaar_dham_activity'] = pd.to_timedelta(required_df['time_spent_in_chaar_dham_activity'])
    required_df['time_spent_in_char_dham_Activity_hours'] = required_df['time_spent_in_chaar_dham_activity'].dt.total_seconds() / 3600.0

    # Group by 'maid' and sum the time spent at Chaar Dham activities
    grouped_df = required_df.groupby(['maid'])['time_spent_in_char_dham_Activity_hours'].sum().reset_index()

    # Calculate and print statistics before removing outliers
    stats_before = grouped_df['time_spent_in_char_dham_Activity_hours'].agg(['mean', 'median', 'std', 'count', 'max', 'min']).reset_index()
    stats_before.columns = ['metric', 'value']

    # Find outliers and filter data
    Q1 = grouped_df['time_spent_in_char_dham_Activity_hours'].quantile(0.25)
    Q3 = grouped_df['time_spent_in_char_dham_Activity_hours'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Identify outliers
    outliers = grouped_df[(grouped_df['time_spent_in_char_dham_Activity_hours'] < lower_bound) |
                          (grouped_df['time_spent_in_char_dham_Activity_hours'] > upper_bound)]
    outlier_count = outliers.shape[0]

    filtered_grouped_df = grouped_df[(grouped_df['time_spent_in_char_dham_Activity_hours'] >= lower_bound) &
                                     (grouped_df['time_spent_in_char_dham_Activity_hours'] <= upper_bound)]

    # Calculate additional statistics for the filtered data
    Q1_after = filtered_grouped_df['time_spent_in_char_dham_Activity_hours'].quantile(0.25)
    Q3_after = filtered_grouped_df['time_spent_in_char_dham_Activity_hours'].quantile(0.75)
    IQR_after = Q3_after - Q1_after
    skewness_after = stats.skew(filtered_grouped_df['time_spent_in_char_dham_Activity_hours'])
    kurtosis_after = stats.kurtosis(filtered_grouped_df['time_spent_in_char_dham_Activity_hours'])
    
    stats_after = filtered_grouped_df['time_spent_in_char_dham_Activity_hours'].agg(['mean', 'median', 'std', 'count', 'max', 'min']).reset_index()
    stats_after.columns = ['metric', 'value']
    
    # Add additional statistics to stats_after
    additional_stats = pd.DataFrame({
        'metric': ['Q1', 'Q3', 'IQR', 'skewness', 'kurtosis', 'outlier_count'],
        'value': [Q1_after, Q3_after, IQR_after, skewness_after, kurtosis_after, outlier_count]
    })
    stats_after = pd.concat([stats_after, additional_stats], ignore_index=True)

    # Plot distribution after removing outliers
    plot_time_spent_in_chardham(filtered_grouped_df, char_dham,  savepath, state)

    return stats_before, stats_after