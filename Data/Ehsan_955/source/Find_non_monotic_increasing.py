import os
import pandas as pd
import glob


# Function to check if a series is monotonically increasing
def is_monotonic_increasing(series):
    return series.is_monotonic_increasing


# Function to process a single CSV file
def check_fuel_consumption(file_path):
    try:
        # Read the CSV file
        df = pd.read_csv(file_path)

        # Check if the required column is in the file
        if 'Trip_fuel_consumption' in df.columns:
            # Check if the column is monotonically increasing
            if not is_monotonic_increasing(df['Trip_fuel_consumption']):
                print(f"File with non-monotonic Trip_fuel_consumption: {file_path}")
        else:
            print(f"Column 'Trip_fuel_consumption' not found in {file_path}")
    except Exception as e:
        print(f"Error reading {file_path}: {e}")


# Function to process all CSV files in the folder and subfolders
def process_folder(folder_path):
    # Get all CSV files recursively
    csv_files = glob.glob(os.path.join(folder_path, '**', '*.csv'), recursive=True)

    # Loop through each file and check fuel consumption
    for file_path in csv_files:
        check_fuel_consumption(file_path)


# Example usage

folder_path = 'C:/Users/s_alizadehnia/Desktop/LSTMPredictions/Data/Ehsan_955/source/processed'  # Replace with your folder path
process_folder(folder_path)
