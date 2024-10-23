import pandas as pd
import geopandas as gpd

from shapely import Point
from pyproj import CRS
from pyproj import Transformer

def extract_region_records(df, region_name, column):
    df= df[df[column]==region_name]
    df = df.sort_values(by = ['maid', 'datetime', 'latitude', 'longitude'])
    df = df.drop_duplicates(subset=['maid', 'datetime', 'latitude', 'longitude'], keep='first')
    return df

def merge_line_shape_file(df, shape_file_path,dist_col, lat_column='latitude', lon_column='longitude'):
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
    if shape_file_path.endswith('railways.shp'):
        shape_gdf = shape_gdf[['name', 'geometry']]
        shape_gdf.rename(columns={'name':'name_rail'}, inplace=True)
    else:
        shape_gdf = shape_gdf[['name','highway','ref_old','ref','geometry']]
        shape_gdf.rename(columns={'name':'name_road'}, inplace=True)
    # Ensure the CRS matches
    gdf.set_crs(epsg=4326, inplace=True)
    gdf = gdf.to_crs("EPSG:3857")
    
    shape_gdf = shape_gdf.to_crs("EPSG:3857")

    # Step 3: Perform a spatial join between the GeoDataFrame and the shapefile GeoDataFrame
    # Using 'inner' join to keep only points that intersect with the polygons in the shapefile
    gdf = gpd.sjoin_nearest(gdf, shape_gdf, how="left", distance_col=dist_col)
    gdf.reset_index(drop=True)

    gdf = gdf.sort_values(by=['maid', 'datetime',lat_column, lon_column, dist_col])
    gdf = gdf.drop_duplicates(subset=['maid', 'datetime', lat_column, lon_column], keep='first')

    # return gdf

    # Step 4: Drop the geometry columns if you want a regular DataFrame after merging
    gdf = pd.DataFrame(gdf.drop(columns=['geometry', 'index_right']))

    return gdf

def merge_airport_shape(df, point_shape_file_path, boundary_shape_file_path, dist_col, lat_column = 'latitude', lon_column= 'longitude'):
    """
    Merges a dataframe containing latitude and longitude information with a shapefile.

    Parameters:
    df (pd.DataFrame): Input dataframe containing 'latitude' and 'longitude' columns (or custom columns defined).
    point_shape_file_path (str): Path to the shapefile to be merged.
    lat_column (str): Name of the latitude column in the dataframe. Defaults to 'latitude'.
    lon_column (str): Name of the longitude column in the dataframe. Defaults to 'longitude'.

    Returns:
    pd.DataFrame: Dataframe merged with the shapefile attributes based on spatial join.
    """
    # Step 1: Read the shapefile into a GeoDataFrame
    point_shape_gdf = gpd.read_file(point_shape_file_path)

    # Step 2: Convert the latitude and longitude columns in the dataframe to a GeoDataFrame
    # Create 'geometry' column by combining the latitude and longitude values into Point objects
    geometry = [Point(xy) for xy in zip(df[lon_column], df[lat_column])]
    
    # Create a GeoDataFrame from the original dataframe
    gdf = gpd.GeoDataFrame(df.copy(), geometry=geometry, crs="EPSG:4326")  # Assuming WGS84 Coordinate Reference System (EPSG:4326)
    # Ensure the CRS matches
    gdf.set_crs(epsg=4326, inplace=True)
    gdf = gdf.to_crs("EPSG:3857")
    
    point_shape_gdf = point_shape_gdf.to_crs("EPSG:3857")
    point_shape_gdf = point_shape_gdf[['name', 'geometry']]
    point_shape_gdf.rename(columns = {'name':'name_point'}, inplace=True)
    point_shape_gdf['name_point'].fillna('Name not available')

    # Step 3: Perform a spatial join between the GeoDataFrame and the shapefile GeoDataFrame
    # Using 'inner' join to keep only points that intersect with the polygons in the shapefile
    gdf = gpd.sjoin_nearest(gdf, point_shape_gdf, how="left", distance_col=dist_col)
    gdf.reset_index(drop=True)
    
    gdf = pd.DataFrame(gdf.drop(columns=['index_right']))
    boundary_shape_gdf = gpd.read_file(boundary_shape_file_path)
    boundary_shape_gdf = boundary_shape_gdf[['osm_id','name', 'geometry', 'aeroway']]
    boundary_shape_gdf.rename(columns = {'name':'name_boundary'}, inplace=True)
    boundary_shape_gdf['name_boundary'].fillna('Name not available')
    boundary_shape_gdf = boundary_shape_gdf.to_crs("EPSG:3857")

    geometry = [Point(xy) for xy in zip(df[lon_column], df[lat_column])]
    
    # Create a GeoDataFrame from the original dataframe
    gdf = gpd.GeoDataFrame(gdf.copy(), geometry=geometry, crs="EPSG:4326")  # Assuming WGS84 Coordinate Reference System (EPSG:4326)
    # Ensure the CRS matches
    gdf.set_crs(epsg=4326, inplace=True)
    gdf = gdf.to_crs("EPSG:3857") 

    gdf = gpd.sjoin(gdf, boundary_shape_gdf, how="left", predicate='within')

    gdf = gdf.sort_values(by=['maid', 'datetime',lat_column, lon_column, dist_col])

    gdf = gdf.drop_duplicates(subset=['maid', 'datetime', lat_column, lon_column], keep='first')

    # Step 4: Drop the geometry columns if you want a regular DataFrame after merging
    gdf = pd.DataFrame(gdf.drop(columns=['geometry', 'index_right']))

    return gdf



# # Projection
# def convert (df):
#     crs_4326 = CRS("WGS84")  # source CRS that is lat and lon
#     crs_proj = CRS("EPSG:3857")    #for Uttarakhand, x and y
#     transformer = Transformer.from_crs(crs_4326, crs_proj)

#     def convert_to_CRS( latitude , longitude ) :
#         y , x =   transformer.transform( latitude , longitude )
#         return ( y,  x )
#     # intrastate_travellers_data["y"] ,intrastate_travellers_data["x"] = convert_to_CRS(intrastate_travellers_data.latitude , intrastate_travellers_data.longitude  )
#     # df_chardham
#     # df_chardham["y"] ,df_chardham["x"] = convert_to_CRS(df_chardham.latitude , df_chardham.longitude  )
#     df["y"] ,df["x"] = convert_to_CRS(df.latitude , df.longitude  )
#     return df

from pyproj import CRS, Transformer
import pandas as pd

# Projection function: convert latitude and longitude to x and y (projected coordinates)
def convert(df):
    crs_4326 = CRS("WGS84")  # source CRS that is lat and lon
    crs_proj = CRS("EPSG:3857")  # target CRS for Uttarakhand, x and y
    transformer = Transformer.from_crs(crs_4326, crs_proj)

    def convert_to_CRS(latitude, longitude):
        y, x = transformer.transform(latitude, longitude)
        return y, x

    # Apply transformation without removing other columns
    df[['y', 'x']] = df.apply(lambda row: pd.Series(convert_to_CRS(row.latitude, row.longitude)), axis=1)
    
    # Return the updated DataFrame with new 'x' and 'y' columns
    return df

# Function to extract indices from the position group string (like 'x_5y_10')
def extract_indices_from_group(group):
    # Extract x_ and y_ from the group string
    x_part, y_part = group.split('y_')
    x_ = int(x_part.replace('x_', ''))
    y_ = int(y_part)
    return x_, y_

# Convert position groups (e.g., 'x_5y_10') to indices
def position_group_to_indices(group):
    x_, y_ = extract_indices_from_group(group)
    return x_, y_

# Function to group the positions into grid cells and assign indices
def make_position_groups(df, grid_size):
    # First, convert the latitude/longitude to projected coordinates (x, y)
    df = convert(df)
    
    # Define minimum x and y to calculate relative grid position
    x_min = df.x.min()
    y_min = df.y.min()

    # Function to calculate the group based on x, y, grid size, and minimum coordinates
    def get_group(x, y, x_min=x_min, y_min=y_min, gap=grid_size):
        x_ = int((x - x_min) / gap)
        y_ = int((y - y_min) / gap)
        group = "x_" + str(x_) + "y_" + str(y_)
        return group

    # Assign position groups based on x and y coordinates, without dropping other columns
    df["position_group"] = df.apply(lambda row: get_group(row.x, row.y), axis=1)
    
    # Extract indices from the position group and add them as new columns 'x_' and 'y_'
    df[['x_', 'y_']] = df['position_group'].apply(lambda grp: pd.Series(position_group_to_indices(grp)))
    
    # Return the DataFrame with all original columns and the new columns 'x', 'y', 'position_group', 'x_', 'y_'
    return df
