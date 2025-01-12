import pandas as pd
import requests
import time
import os
from tqdm import tqdm

# Global variable to store the last received altitude
last_received_altitude = None

def get_altitude(lat, lon, retries=10):
    """
    Fetches altitude data from an API for a given latitude and longitude.
    Uses the last successful altitude as a fallback if API calls fail.

    Parameters:
    - lat (float): Latitude.
    - lon (float): Longitude.
    - retries (int): Number of retry attempts in case of failure.

    Returns:
    - float: Altitude value or None if unavailable.
    """
    global last_received_altitude  # Use the last received altitude as fallback
    url = f'https://www.elevation-api.eu/v1/elevation?pts=[[{lat},{lon}]]'
    for attempt in range(retries):
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                elevation_data = response.json()
                if 'elevations' in elevation_data and elevation_data['elevations']:
                    altitude = elevation_data['elevations'][0]  # Extract the first elevation value
                    last_received_altitude = altitude  # Update last successful altitude
                    return altitude
            elif response.status_code == 504:
                return last_received_altitude  # Use last received altitude on 504 error
        except requests.exceptions.RequestException:
            time.sleep(2)  # Wait for 2 seconds before retrying
    return last_received_altitude  # Use last received altitude if all retries fail

def get_altitude_for_block(df, start_index , selected_columns):
    """
    Fetches altitude for a block of rows in a DataFrame.

    Parameters:
    - df (pd.DataFrame): DataFrame containing selected_columns["latitude_column"] and selected_columns["longitude_column"] columns.
    - start_index (int): Starting index of the block.

    Returns:
    - float: Altitude value for the block or None.
    """
    for i in range(start_index, min(start_index + 100, len(df))):
        lat = df.at[i, selected_columns["latitude_column"]]
        lon = df.at[i, selected_columns["longitude_column"]]

        # Skip rows with missing or invalid latitude/longitude
        if pd.isna(lat) or pd.isna(lon) or lat == 0 or lon == 0:
            continue

        altitude = get_altitude(lat, lon)
        if altitude is not None:
            return altitude
    return None

def process_altitudes(folder_path , selected_columns):
    """
    Processes altitude data for each CSV file in a specified folder.
    Adds a 'correct_altitude' column to the CSV files.

    Parameters:
    - folder_path (str): Path to the folder containing CSV files.

    Returns:
    - None
    """
    for filename in tqdm(os.listdir(folder_path), desc="Processing files"):
        if filename.endswith('.csv'):
            file_path = os.path.join(folder_path, filename)
            data_df = pd.read_csv(file_path)

            # Check if selected_columns["latitude_column"] and selected_columns["longitude_column"] columns exist in the dataframe
            if selected_columns["latitude_column"] not in data_df.columns or selected_columns["longitude_column"] not in data_df.columns:
                continue  # Skip this file if columns are missing

            # Initialize the 'correct_altitude' column with None values
            data_df['correct_altitude'] = None

            # Process rows in blocks of 100
            for i in range(0, len(data_df), 100):
                altitude = get_altitude_for_block(data_df, i , selected_columns)

                # Adjust the end of the range to ensure it's within bounds and handle small blocks
                block_end = min(i + 99, len(data_df) - 1)

                # If the altitude is None and the block size is 1 (e.g., last row), skip setting altitude
                if altitude is None and (block_end == i):
                    continue

                # Set the altitude for the current block
                data_df.loc[i:block_end, 'correct_altitude'] = altitude

            # Save the updated dataframe to a new CSV file, retaining the same name
            output_file_path = os.path.join(folder_path, filename)  # Save to the same file
            data_df.to_csv(output_file_path, index=False)

    print("Altitude processing completed successfully.")
