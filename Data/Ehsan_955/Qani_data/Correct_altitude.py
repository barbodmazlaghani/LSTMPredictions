 # Update to your folder path
import pandas as pd
import requests
import time
import os

# Path to the folder containing CSV files
folder_path = "C:/Users/s_alizadehnia/Desktop/LSTMPredictions/Data/Ehsan_955/Qani_data/"  # Update to your folder path

# Initialize last received altitude
last_received_altitude = None

# Function to get altitude
def get_altitude(lat, lon, retries=10):
    global last_received_altitude  # Use the last received altitude as fallback
    url = f"https://api.open-elevation.com/api/v1/lookup?locations={lat},{lon}"

    for attempt in range(retries):
        try:
            print("TRY", lat, lon)
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                elevation_data = response.json()
                if elevation_data['results']:
                    altitude = elevation_data['results'][0]['elevation']
                    print(f"Received altitude: {altitude}")
                    last_received_altitude = altitude  # Update last successful altitude
                    return altitude
            elif response.status_code == 504:
                print("504 Gateway Timeout, using last received altitude:", last_received_altitude)
                return last_received_altitude  # Use last received altitude on 504 error
            else:
                print(f"Unexpected status code {response.status_code}: {response.text}")
        except requests.exceptions.RequestException as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            time.sleep(2)  # Wait for 2 seconds before retrying
    print(f"Failed to get altitude for Lat: {lat}, Lon: {lon} after {retries} retries. Using last received altitude: {last_received_altitude}")
    return last_received_altitude  # Use last received altitude if all retries fail

# Helper function to get altitude for a block of 100 rows
def get_altitude_for_block(df, start_index):
    for i in range(start_index, min(start_index + 100, len(df))):
        lat = df.at[i, 'latitude']
        lon = df.at[i, 'longitude']

        # Skip rows with missing or invalid latitude/longitude
        if pd.isna(lat) or pd.isna(lon) or lat == 0 or lon == 0:
            print(f"Skipping empty or invalid row at index {i}")
            continue

        altitude = get_altitude(lat, lon)
        if altitude is not None:
            return altitude
    return None

# Process each CSV file in the specified folder
for filename in os.listdir(folder_path):
    if filename.endswith('.csv'):
        print(f"Processing file: {filename}")
        file_path = os.path.join(folder_path, filename)
        data_df = pd.read_csv(file_path)

        # Check if 'latitude' and 'longitude' columns exist in the dataframe
        if 'latitude' not in data_df.columns or 'longitude' not in data_df.columns:
            print(f"Skipping file {filename} due to missing 'latitude' or 'longitude' columns.")
            continue  # Skip this file if columns are missing

        # Initialize the 'correct_altitude' column with None values
        data_df['correct_altitude'] = None

        # Get the correct altitudes for each block of 100 rows
        for i in range(0, len(data_df), 100):
            altitude = get_altitude_for_block(data_df, i)

            # Adjust the end of the range to ensure it's within bounds and handle small blocks
            block_end = min(i + 99, len(data_df) - 1)

            # If the altitude is None and the block size is 1 (e.g., last row), skip setting altitude
            if altitude is None and (block_end == i):
                print(f"Skipping last single-row block at index {i} with no valid data.")
                continue

            # Set the altitude for the current block
            data_df.loc[i:block_end, 'correct_altitude'] = altitude

        # Save the updated dataframe to a new CSV file, retaining the same name
        output_file_path = os.path.join(folder_path, filename)  # Save to the same file
        data_df.to_csv(output_file_path, index=False)

print("Updated files saved successfully.")
