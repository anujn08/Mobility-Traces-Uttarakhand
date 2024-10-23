import os
import pandas as pd
import time
from tqdm import tqdm
import json

# Function to trace all visitors from a single file
def trace_all_visitors(input_path, visitors):
    start_time = time.time()  # Record start time

    # Load data into a Pandas DataFrame
    print(f'\nInput file: {input_path}')
    data = pd.read_csv(input_path)
    data_shape = data.shape[0]
    print(f'\nShape of file: {data.shape}')

    # Filter out records of visitors who visited Uttarakhand
    data = data[data['DEVICE_ID'].isin(visitors)]
    print(f'\nShape of file after filtering on basis of Uttarakhand visit: {data.shape}')

    end_time = time.time()  # Record end time
    elapsed_time = end_time - start_time  # Calculate elapsed time
    print(f"\nTime taken for reading {input_path} data with {data_shape} records: {elapsed_time} seconds")
    print('\nPROCESSED THE FILE______')

    # Return relevant columns
    data = data[['DEVICE_ID', 'LATITUDE', 'LONGITUDE', 'EVENT_DATE']]
    data['EVENT_DATE'] = pd.to_datetime(data['EVENT_DATE'])
    return data[['DEVICE_ID', 'LATITUDE', 'LONGITUDE', 'EVENT_DATE']]

# Main function to process all files and extract Uttarakhand visitors
def lookup_visitors(input_dir, visitors):
    columns = ['DEVICE_ID', 'LATITUDE', 'LONGITUDE', 'EVENT_DATE']
    part_dirs = [d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))]

    for part_dir in part_dirs:
        print(f'\nProcessing directory: {part_dir}')
        part_path = os.path.join(input_dir, part_dir)
        files_path = [os.path.join(part_path, f) for f in os.listdir(part_path) if f.endswith('.csv')]

        total_visitors = pd.DataFrame(columns=columns)
        count = len(files_path)
        print(f'Total number of files in {part_dir}: {count}')

        for itr in tqdm(range(count), desc=f"Processing files in {part_dir}"):
            temp_df = trace_all_visitors(files_path[itr], visitors)
            total_visitors = pd.concat([total_visitors, temp_df])

            print(f'\nTotal visitors shape: {total_visitors.shape}')

        file_save_path = os.path.join('', 'filtered_records')
        os.makedirs(file_save_path, exist_ok=True)
        output_file_path = os.path.join(file_save_path, f'total_visitors_{part_dir}.csv')
        total_visitors.to_csv(output_file_path, index=False)
        print(f'Data saved as: {output_file_path}')

if __name__ == "__main__":
    # Example usage
    input_dir = r'C:\Users\Admin\Desktop\Desktop\Mobility_traces_code\output_dir_unzipped_csv'  # Replace with your actual input directory

    # Assuming visitors is defined somewhere
    with open(r'C:\Users\Admin\Desktop\Desktop\Mobility_traces_code\JSON_files\List of Visitors\Distinct_visitors_upto_3.json', 'r') as f:
        visitors = json.load(f)
    visitors = set(visitors)

    lookup_visitors(input_dir, visitors)

# import os
# import pandas as pd
# import time
# from tqdm import tqdm
# import json

# # Function to trace all visitors from a single file
# def trace_all_visitors(input_path, visitors):
#     start_time = time.time()  # Record start time

#     # Load data into a Pandas DataFrame
#     print(f'\nInput file: {input_path}')
#     data = pd.read_csv(input_path)
#     data_shape = data.shape[0]
#     print(f'\nShape of file: {data.shape}')

#     # Filter out records of visitors who visited Uttarakhand
#     data = data[data['DEVICE_ID'].isin(visitors)]
#     print(f'\nShape of file after filtering on basis of Uttarakhand visit: {data.shape}')

#     end_time = time.time()  # Record end time
#     elapsed_time = end_time - start_time  # Calculate elapsed time
#     print(f"\nTime taken for reading {input_path} data with {data_shape} records: {elapsed_time} seconds")
#     print('\nPROCESSED THE FILE______')

#     # Return relevant columns
#     data = data[['DEVICE_ID', 'LATITUDE', 'LONGITUDE', 'EVENT_DATE']]
#     data['EVENT_DATE']= pd.to_datetime(data['EVENT_DATE'])
#     return data[['DEVICE_ID', 'LATITUDE', 'LONGITUDE', 'EVENT_DATE']]

# # Main function to process all files and extract Uttarakhand visitors
# def lookup_visitors(input_dir, files_path, visitors):
#     columns = ['DEVICE_ID', 'LATITUDE', 'LONGITUDE', 'EVENT_DATE']
#     total_visitors = pd.DataFrame(columns=columns)
#     flag = True
#     count = len(files_path)
#     print(f'Total number of files: {count}')

#     for itr in tqdm(range(count), desc="Processing files to trace all visitors data"):
#         temp_df = trace_all_visitors(files_path[itr], visitors)
#         total_visitors = pd.concat([total_visitors, temp_df])

#         # if (itr + 1) % 16 == 0:
#             # subset_number = (itr + 1) // 16
#         print(f'\nTotal visitors shape: {total_visitors.shape}')
#         # file_save_path = f'/content/drive/MyDrive/Sample_Traces_data/Filtered_records_for_New_Data_files'
#         # os.makedirs(file_save_path, exist_ok=True)
#         # total_visitors.to_csv(f'{file_save_path}/filtered_data_{itr}.csv', index=False)
#         # print(f'Data saved as: {file_save_path}/filtered_data_{subset_number}.csv')
#         # total_visitors = pd.DataFrame(columns=columns)

#     # if flag:
#         # subset_number = (itr + 1) // 16 + 1
#     file_save_path = r'C:\Users\Admin\Desktop\Desktop\Mobility_traces_code\filtered_records'
#     os.makedirs(file_save_path, exist_ok=True)
#     total_visitors.to_csv(f'{file_save_path}/filtered_data_.csv', index=False)
#     print(f'Data saved as: {file_save_path}/filtered_data_.csv')

# if __name__ == "__main__":
#     # Example usage
#     input_dir = r'C:\Users\Admin\Desktop\Desktop\Mobility_traces_code\output_dir_unzipped_csv'  # Replace with your actual input directory
#     files_path = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith('.csv')]

#     # Assuming visitors is defined somewhere
#     # visitors = {'maid1': 'value1', 'maid2': 'value2'}  # Replace with your actual dictionary
#     # visitors=[]
#     with open(r'C:\Users\Admin\Desktop\Desktop\Mobility_traces_code\Mobility_traces_json_unzipped_files_bkp\List of Visitors\Distinct_visitors_upto_3.json', 'r') as f:
#         visitors =json.load(f)
#     visitors=set(visitors)

#     lookup_visitors(files_path, visitors)
