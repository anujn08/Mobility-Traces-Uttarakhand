
import os
import json
import argparse
import warnings
import geopandas as gpd
from tqdm import tqdm
from loading_files.files_utils import get_files
from loading_files.geometry_utils import load_geometry
from loading_files.trace_utils import is_within_shape, process_trace_data
from lookup_visitors.looking_up_visitors import get_last_json_file, lookup_visitors # (input_path, visitors)
from loading_files.get_paths import get_csv_paths

# Settings the warnings to be ignored
warnings.filterwarnings('ignore')



def process_each_file(sample_dir, geojson_file, output_dir_json, output_dir_unzipped_csv, latitude_min, latitude_max , longitude_min, longitude_max ):
    # Ensure the script's directory is the current working directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    # Create the output directory in the root directory for saving unzipped files
    # output_dir_unzipped_csv = os.path.join(script_dir, output_dir_unzipped_csv)
    os.makedirs(output_dir_unzipped_csv, exist_ok=True)
    
    # Create the output directory in the root directory for saving json files
    # output_dir_json = os.path.join(script_dir, output_dir_json)
    os.makedirs(output_dir_json, exist_ok=True)

    # Load the geometry
    shape_uttarakhand = load_geometry(geojson_file)
    
    # Get the files
    files_path = get_files(sample_dir, output_dir_unzipped_csv, doc_format="gz")
    # files_path = get_csv_paths(sample_dir)
    # print(f'files_path = {files_path}')
    # Initialize visitors set
    visitors = set()
    count = len(files_path)
    print(f'\n\n Count: {count}')
    
    # Process each file
    for itr in tqdm(range(count)):
        visitors = process_trace_data(
            files_path[itr],
            output_dir_unzipped_csv,
            output_dir_json,
            visitors,
            itr,
            chunk_size=10000000,
            shape_file=shape_uttarakhand,
            latitude= [latitude_min, latitude_max],
            longitude=[longitude_min, longitude_max]
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process mobility trace data and save results.")
    parser.add_argument('sample_dir', type=str, help='The directory containing the sample .gz files')
    parser.add_argument('geojson_file', type=str, help='The path to the GeoJSON file')
    parser.add_argument('output_dir_json', type=str, help='The output directory to save results')
    parser.add_argument('output_dir_unzipped_csv', type=str, help='The output directory to save results')
    parser.add_argument('output_dir_filtered_csv', type=str, help='The output directory to save filtered records')
    parser.add_argument('latitude_min', type=float, help='narrows down the search algorithm by filtering out the records falling outside this range')
    parser.add_argument('latitude_max', type=float, help='narrows down the search algorithm by filtering out the records falling outside this range')
    parser.add_argument('longitude_min', type=float, help='narrows down the search algorithm by filtering out the records falling outside this range')
    parser.add_argument('longitude_max', type=float, help='narrows down the search algorithm by filtering out the records falling outside this range')
    # parser.add_argument('longitude_min_max_list', type=list, help='narrows down the search algorithm by filtering out the records falling outside this range')
    args = parser.parse_args()
    
    process_each_file(args.sample_dir, args.geojson_file, args.output_dir_json, args.output_dir_unzipped_csv,\
                      args.latitude_min, args.latitude_max, args.longitude_min, args.longitude_max)
    
    visitors= get_last_json_file(args.output_dir_json)

    visitors=set(visitors)
    print(f'Count of visitors : {len(visitors)}')

    # print(len(visitors))
    lookup_visitors(args.output_dir_unzipped_csv,visitors, args.output_dir_filtered_csv)
