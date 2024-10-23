
# README: Mobility Trace Data Processing Script

## Overview

This Python script processes mobility trace data from `.gz` files, filters records based on geographical boundaries (defined in a GeoJSON file), and saves the results in specified directories. The script handles large data by processing the trace files in chunks and filtering them based on latitude and longitude ranges.

### Key Features:
- Filters mobility traces based on latitude and longitude boundaries.
- Unzips and processes `.gz` files in chunks to efficiently handle large datasets.
- Loads geographical boundaries from a GeoJSON file to filter traces that fall within the defined region.
- Saves the filtered results and visitors' data in JSON and CSV formats.

---

## Dependencies

The script requires the following Python packages:

- `os`: For file and directory operations.
- `json`: For handling JSON files.
- `argparse`: For parsing command-line arguments.
- `geopandas` (`gpd`): For handling geospatial data and loading GeoJSON files.
- `tqdm`: For displaying progress bars during data processing.
- Custom modules:
  - `loading_files.files_utils`: Functions for file management (`get_files`).
  - `loading_files.geometry_utils`: Handles geometry-related functions (`load_geometry`).
  - `loading_files.trace_utils`: Processes trace data and checks for boundary conditions (`process_trace_data`).
  - `lookup_visitors.looking_up_visitors`: Functions for managing visitors and loading JSON files.
  - `loading_files.get_paths`: Retrieves CSV file paths.

Ensure all dependencies are installed using pip before running the script:
```bash
pip install geopandas tqdm
```

---

## Usage

To run the script, you need to provide the following arguments:

```bash
python script_name.py <sample_dir> <geojson_file> <output_dir_json> <output_dir_unzipped_csv> <output_dir_filtered_csv> <latitude_min> <latitude_max> <longitude_min> <longitude_max>
```

### Positional Arguments:

1. **`sample_dir`**: Directory containing the sample `.gz` files (input data).
2. **`geojson_file`**: Path to the GeoJSON file defining the geographical boundaries.
3. **`output_dir_json`**: Directory to save visitors' data in JSON format.
4. **`output_dir_unzipped_csv`**: Directory to save unzipped `.csv` files.
5. **`output_dir_filtered_csv`**: Directory to save filtered CSV files with visitor data.
6. **`latitude_min`**: Minimum latitude for filtering records.
7. **`latitude_max`**: Maximum latitude for filtering records.
8. **`longitude_min`**: Minimum longitude for filtering records.
9. **`longitude_max`**: Maximum longitude for filtering records.

### Example Command:

```bash
python process_trace_data.py ./data ./uttarakhand_boundary.geojson ./output_json ./output_csv ./filtered_csv 29.0 30.5 78.0 80.5
```

This command processes the trace data from the `.gz` files in the `./data` directory, filters them using the specified latitude and longitude bounds, and saves the results in the respective output directories.

---

## Output Files

1. **Unzipped CSV Files**: These files contain unzipped data from the `.gz` trace files.
2. **JSON Files**: These files contain visitors' data, representing traces within the defined geographical boundaries.
3. **Filtered CSV Files**: These files contain records filtered by visitor data and geographical bounds.

