
# README

## Overview

This script takes the final processed CSV file and then merge it with the shape files. It merges the file with India Shape file, region shape file (example Uttarakhand here), regions road network shape files, rail network shape files and airport/helipad shape files.

### Key Steps:
1. After merging with India Shape file, it makes OD matrices.
2. After merging with region shape files and respective, road, rail, airport shape files, it determines the activity performed by the individual i.e. travelling by road, rail or airways.
3. For Uttarakhand, it also creates a point shape file for the important locations for char dham visitors, and it is determined how many people, for how long, from what time to what time stayed at these locations. Also it is found out that which location is visited first. A proper distribution of time spent at each location is found out.


### Environment is already created.


## Arguments

The script accepts the following arguments:

1. --csv (type: str, required: True) 
   - Description: File path of the CSV file.

2. --output (type: str, required: True) 
   - Description: Output directory path for the processed files.

3. --India_shape_file (type: str, required: True) 
   - Description: Path to the India shapefile.

4. --region_name (type: str, optional) 
   - Description: Region name in short form.

5. --road_shape_file (type: str, optional) 
   - Description: Path to the road shapefile (LINE SHAPE).

6. --rail_shape_file (type: str, optional) 
   - Description: Path to the rail shapefile (LINE SHAPE).

7. --airport_point_shape_file (type: str, optional) 
   - Description: Path to the airport point shapefile (LINE SHAPE).

8. --airport_boundary_shape_file (type: str, optional) 
   - Description: Path to the airport boundary shapefile (LINE SHAPE).


## Running the Script

You can run the script using the following command:

sample running script:

C:\Users\User\Downloads\Mobility_traces_new_data_code\mobility_traces_env\Scripts\python.exe C:\Users\User\Downloads\mobility_traces_complete_data\main_part_3_joining_shape_files_OD.py --csv "C:\Users\User\Downloads\mobility_traces_uttarakhand\df_all_valid_distributed_IDs_data.csv" --output "C:\Users\User\Downloads\mobility_traces_uttarakhand\with_shape_files_merged" --India_shape_file "C:\Users\User\Downloads\Mobility_traces_new_data_code\GeoJSON files\India_Districts_map_JK_LD.geojson" --region_name "UK" --road_shape_file "C:\Users\User\Downloads\mobility_traces_complete_data\shape file for roads and railway network\QuickOSM generated\UK_roads_all_type\Uttarakhand_roads_all_type.shp" --rail_shape_file "C:\Users\User\Downloads\mobility_traces_complete_data\shape file for roads and railway network\QuickOSM generated\UK_rail_network\Uttarakhand_railways.shp" --airport_point_shape_file "C:\Users\User\Downloads\mobility_traces_complete_data\shape file for roads and railway network\QuickOSM generated\UK_airport\Uttarakhand_airports_points.shp" --airport_boundary_shape_file "C:\Users\User\Downloads\mobility_traces_complete_data\shape file for roads and railway network\QuickOSM generated\UK_airport\Uttarakhand_airports_boundary.shp"