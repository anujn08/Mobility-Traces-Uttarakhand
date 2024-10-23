
# README

## Overview

This script processes and combines multiple CSV files, performs velocity computations, and applies various filtering steps. It is designed to handle large datasets efficiently and remove unnecessary records based on specific conditions.

### Key Steps:
1. Combines all CSV files from the specified directory.
2. Converts latitude and longitude coordinates to a new CRS (Coordinate Reference System).
3. Computes velocity and displacement values for each record.
4. Applies Type 1, Type 2, and Type 3 removals based on custom logic to clean the dataset.
5. Filters out records that do not meet the required conditions for validity.
6. Saves the processed data into output CSV files at each step.

## Dependencies

To run this script, ensure you have the following Python packages installed:

- `pandas`
- `argparse`
- `dask`
- `tqdm`
- Custom modules:
    - `dist_velocity_type_1_type_2_computation`
    - `combine_all_filtered_csvs`
    - `remove_unnecessary_records`

You can install the required dependencies using the following command:

```
pip install pandas dask tqdm
```

## Arguments

The script accepts the following arguments:

- `--csv_dir`: The directory containing CSV files to combine.
- `--output_csv`: The output directory for the combined CSV file.
- `--type_1_save_path`: The path where Type 1 removal results will be saved.
- `--type_2_save_path`: The path where Type 2 removal results will be saved.

## Running the Script

You can run the script using the following command:


python script_name.py --csv_dir /path/to/csv_directory --output_csv /path/to/output_directory --type_1_save_path /path/to/type1_output --type_2_save_path /path/to/type2_output
