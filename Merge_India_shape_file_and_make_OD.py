import geopandas as gpd
import pandas as pd
from shapely.geometry import Point
from tqdm import tqdm

import pandas as pd
from datetime import datetime, timedelta
import os

def merge_with_codes_and_create_path(df, state_column='st_nm', district_column='district'):
    """
    Merges the input DataFrame with predefined state and district short codes, and creates a 'path' column based on 
    specific rules for each state.

    Parameters:
    df (pd.DataFrame): Input dataframe containing 'state' and 'district' columns (or custom column names defined).
    state_column (str): The column name representing state information in the input DataFrame. Defaults to 'state'.
    district_column (str): The column name representing district information in the input DataFrame. Defaults to 'district'.

    Returns:
    pd.DataFrame: DataFrame with 'state_short', 'district_short', and 'path' columns.
    """
    # Define the state and RTO codes mapping
    state_rto_data = {
        "State/UT": [
            "Andhra Pradesh", "Arunachal Pradesh", "Assam", "Bihar",
            "Chhattisgarh", "Goa", "Gujarat", "Haryana",
            "Himachal Pradesh", "Jharkhand", "Karnataka", "Kerala",
            "Madhya Pradesh", "Maharashtra", "Manipur", "Meghalaya",
            "Mizoram", "Nagaland", "Odisha", "Punjab",
            "Rajasthan", "Sikkim", "Tamil Nadu", "Telangana",
            "Tripura", "Uttar Pradesh", "Uttarakhand", "West Bengal",
            "Andaman and Nicobar", "Chandigarh", "Dadra and Nagar Haveli",
            "Lakshadweep", "Delhi", "Puducherry", "Jammu and Kashmir","JAMMU & KASHMIR", "Ladakh"
        ],
        "state_short": [
            "AP", "AR", "AS", "BR",
            "CG", "GA", "GJ", "HR",
            "HP", "JH", "KA", "KL",
            "MP", "MH", "MN", "ML",
            "MZ", "NL", "OD", "PB",
            "RJ", "SK", "TN", "TG",
            "TR", "UP", "UK", "WB",
            "AN", "CH", "DD",
            "LD", "DL", "PY", "JK","JK", "LA"
        ]
    }
    
    # Define the district and short code mapping for Uttarakhand
    district_short_names = {
        "District Name": [
            "Almora", "Bageshwar", "Chamoli", "Champawat", "Dehradun",
            "Haridwar", "Nainital", "Pauri Garhwal", "Pithoragarh",
            "Rudraprayag", "Tehri Garhwal", "Udham Singh Nagar", "Uttarkashi"
        ],
        "district_short": [
            "ALM", "BGR", "CHM", "CHP", "DDN", "HWR", "NTL", "PRG",
            "PTH", "RUD", "TRG", "USN", "UTK"
        ]
    }
    
    # Create DataFrames for state and district mappings
    df_rto_codes = pd.DataFrame(state_rto_data)
    df_district_short = pd.DataFrame(district_short_names)

    # Merge the input dataframe with the state codes
    df = pd.merge(df, df_rto_codes, how='left', left_on=state_column, right_on='State/UT')
    
    # Merge the dataframe with the district codes for Uttarakhand
    df = pd.merge(df, df_district_short, how='left', left_on=district_column, right_on='District Name')
    
    # Create the 'path' column based on state and district codes
    df['path'] = df.apply(lambda row: row['district_short'] if row[state_column] == 'Uttarakhand' else row['state_short'], axis=1)
    # Create 'origin_state' and 'origin_district' based on the first occurrence for each 'id'
    df = df.sort_values(by = ['maid', 'datetime', 'latitude', 'longitude'])
    origin_df = df.groupby('maid').first().reset_index()
    origin_df = origin_df[['maid', 'state_short', district_column]]
    origin_df.rename(columns={'state_short': 'origin_state', district_column: 'origin_district'}, inplace=True)

    # Merge the origin columns back to the main dataframe
    df = pd.merge(df, origin_df, on='maid', how='left')

    # Drop unnecessary columns if needed
    df.drop(columns=['State/UT', 'District Name'], inplace=True)
    df.rename(columns={'state_short': 'state'}, inplace=True)
    
    return df



def merge_shape_file(df, shape_file_path, lat_column='latitude', lon_column='longitude'):
    """
    Merges a dataframe containing latitude and longitude information with a shapefile.

    Parameters:
    df (pd.DataFrame): Input dataframe containing 'latitude' and 'longitude' columns (or custom columns defined).
    shape_file_path (str): Path to the shapefile to be merged.
    lat_column (str): Name of the latitude column in the dataframe. Defaults to 'latitude'.
    lon_column (str): Name of the longitude column in the dataframe. Defaults to 'longitude'.

    Returns:
    pd.DataFrame: Dataframe merged with the shapefile attributes based on spatial join.
    """
    # Step 1: Read the shapefile into a GeoDataFrame
    shape_gdf = gpd.read_file(shape_file_path)

    # Step 2: Convert the latitude and longitude columns in the dataframe to a GeoDataFrame
    # Create 'geometry' column by combining the latitude and longitude values into Point objects
    geometry = [Point(xy) for xy in zip(df[lon_column], df[lat_column])]
    
    # Create a GeoDataFrame from the original dataframe
    gdf = gpd.GeoDataFrame(df.copy(), geometry=geometry, crs="EPSG:4326")  # Assuming WGS84 Coordinate Reference System (EPSG:4326)

    # Ensure the CRS matches
    gdf.set_crs(epsg=4326, inplace=True)
    shape_gdf = shape_gdf.to_crs(epsg=4326)

    # Step 3: Perform a spatial join between the GeoDataFrame and the shapefile GeoDataFrame
    # Using 'inner' join to keep only points that intersect with the polygons in the shapefile
    merged_gdf = gpd.sjoin(gdf, shape_gdf[['district','st_nm', 'geometry']], how="left", predicate="intersects")

    # Step 4: Drop the geometry columns if you want a regular DataFrame after merging
    merged_df = pd.DataFrame(merged_gdf.drop(columns=['geometry', 'index_right']))
    merged_df = merge_with_codes_and_create_path(merged_df)

    return merged_df

import pandas as pd

def extract_origin_info(df, maid_column='maid', state_column='st_nm', district_column='district'):
    """
    Extracts the origin state and district for each unique 'maid' based on the very first record.

    Parameters:
    df (pd.DataFrame): Input dataframe containing information of 'maid', 'state', and 'district'.
    maid_column (str): Column name representing 'maid' in the dataframe. Defaults to 'maid'.
    state_column (str): Column name representing state information in the dataframe. Defaults to 'state'.
    district_column (str): Column name representing district information in the dataframe. Defaults to 'district'.

    Returns:
    pd.DataFrame: New dataframe containing 'maid', 'origin_state', and 'origin_district' based on the first occurrence.
    """
    # Sort the dataframe by 'maid' and any time-related column to ensure we capture the first record correctly
    df_sorted = df.sort_values(by=[maid_column, 'datetime'])  # Adjust 'datetime' if needed based on your time column name
    print(f'columns:{df_sorted.columns}')
    # Drop duplicate rows based on the 'maid' column to get the first occurrence of each maid
    df_first_occurrence = df_sorted.drop_duplicates(subset=[maid_column], keep='first')
    df_first_occurrence.rename(columns = {state_column: 'origin_state', district_column:'origin_district'}, inplace=True)
    # # Create a new dataframe with only 'maid', 'origin_state', and 'origin_district' columns
    # df_origin_info = df_first_occurrence[[maid_column, state_column, district_column]].copy()
    # df_origin_info.columns = [maid_column, 'origin_state', 'origin_district']  # Rename columns for clarity

    return df_first_occurrence



def plot_trip_chain(df, prefix=''):
    """
    Plots the cumulative distance against cumulative time for each maid in the dataframe.

    Parameters:
    df (pd.DataFrame): DataFrame containing maid, datetime, distance, and path.
    """

    # Check and convert the datetime column if necessary
    if not pd.api.types.is_datetime64_any_dtype(df['datetime']):
        df['datetime'] = pd.to_datetime(df['datetime'])

    # Plotting cumulative distance against cumulative time
    plt.figure(figsize=(18, 6))

    for maid, group in df.groupby('maid'):
        plt.plot(group['datetime'], group['distance'], marker='o', linestyle='-', label=maid)
        
        # Annotate each point with the corresponding path value
        for i, row in group.iterrows():
            plt.annotate(
                row['path'], 
                (row['datetime'], row['distance']),
                textcoords="offset points", 
                xytext=(5, 5),  # Position of text relative to the point
                ha='left', 
                fontsize=6, 
                alpha=0.7,  # Make the text slightly transparent for clarity
                bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', boxstyle='round,pad=0.3')
            )

    plt.xlabel('Date')
    plt.ylabel('Distance (Kms)')

    # Adjusting the legend position and appearance
    plt.legend(
        loc='upper left',
        title='Maid ID',
        fancybox=True,
        framealpha=0.5,
        borderpad=1
    )

    plt.title(f'{prefix}Distance vs Time')

    # Set the x-axis major locator to day and format the dates
    ax = plt.gca()
    ax.xaxis.set_major_locator(mdates.DayLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    # ax.xaxis.set_minor_locator(mdates.HourLocator(interval=1))
    # ax.xaxis.set_minor_formatter(mdates.DateFormatter('%H:%M'))

    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# Modify the district_OD_matrix function
def district_OD_matrix(visitors, start_date, end_date):
    """
    Summarizes the direct travels between districts within a given date range.

    Parameters:
    visitors (pd.DataFrame): The dataframe containing visitor records.
    start_date (datetime.date): The start date.
    end_date (datetime.date): The end date.

    Returns:
    pd.DataFrame: A summary dataframe showing the count of travels traveling from one district to another.
    """
    # Ensure datetime column is in datetime format
    visitors['datetime'] = pd.to_datetime(visitors['datetime'], format='%Y-%m-%d %H:%M:%S')

    # Extract date part (ensure it is of type datetime.date)
    visitors['date'] = visitors['datetime'].dt.date

    # Replace None or NaN values in the 'district_short' column with 'Outside'
    visitors['district_short'] = visitors['district_short'].fillna('Outside')
    visitors['prev_district_name'] = visitors.groupby('maid')['district_short'].shift(1)

    # Convert start_date and end_date to datetime.date for compatibility
    start_date = pd.to_datetime(start_date).date()
    end_date = pd.to_datetime(end_date).date()

    # Filter the dataframe based on the date range using date objects
    filtered_visitors = visitors[(visitors['date'] >= start_date) & (visitors['date'] <= end_date)]

    if filtered_visitors.empty:
        print("No records found within the given date range.")
        return pd.DataFrame(columns=['Travelled from', 'Travelled to', 'Count of trips'])

    # Identify changes in district
    filtered_visitors['changed_district'] = filtered_visitors['district_short'] != filtered_visitors['prev_district_name']

    # Filter only the rows where district has changed and the previous district is not NaN
    visitors_changes = filtered_visitors[filtered_visitors['changed_district'] & filtered_visitors['prev_district_name'].notna()]

    if visitors_changes.empty:
        print("No district changes found within the given date range.")
        return pd.DataFrame(columns=['Travelled from', 'Travelled to', 'Count of trips'])

    # Group by 'Travelled from' and 'Travelled to' and count number of transitions
    travel_summary = visitors_changes.groupby(['prev_district_name', 'district_short']).size().reset_index(name='Count')

    # Rename columns for clarity
    travel_summary.columns = ['Travelled from', 'Travelled to', 'Count of trips']

    # Create OD matrix
    od_matrix = pd.pivot_table(travel_summary, values='Count of trips', index='Travelled from', columns='Travelled to', fill_value=0)

    print("Origin-Destination Matrix:")
    return od_matrix


# Modify the district_OD_matrix_datewise function
def district_OD_matrix_datewise(visitors, start_date, end_date, file_save_dir):
    """
    Generates the OD matrix for each date in the given date range and saves it as a CSV.

    Parameters:
    visitors (pd.DataFrame): The dataframe containing visitor records.
    start_date (str or datetime-like): The start date in 'YYYY-MM-DD' format or datetime-like.
    end_date (str or datetime-like): The end date in 'YYYY-MM-DD' format or datetime-like.
    file_save_dir (str): The directory path to save the CSV files.
    """
    # Ensure the save directory exists
    if not os.path.exists(file_save_dir):
        os.makedirs(file_save_dir, exist_ok=True)

    # Convert start and end dates to datetime.date type for compatibility
    current_date = pd.to_datetime(start_date).date()
    end_date = pd.to_datetime(end_date).date()

    while current_date <= end_date:
        # Generate the OD matrix for the current date
        od_data = district_OD_matrix(visitors, current_date, current_date)
        
        # Convert the resulting OD matrix to a DataFrame
        df = pd.DataFrame(od_data)

        # Fill NaN values with 0 (if necessary)
        df.fillna(0, inplace=True)

        # Define the CSV filename using only the date part

        file_name = f"district_OD_Matrix_{current_date}.csv"

        # Save the DataFrame to a new CSV file
        file_save_path = os.path.join(file_save_dir, file_name)
        df.to_csv(file_save_path, index=False)

        print(f"Data saved for OD_matrix: {file_save_path}")

        # Move to the next date
        current_date += timedelta(days=1)

    print("Data saved for all OD matrices datewise in CSV format.")

