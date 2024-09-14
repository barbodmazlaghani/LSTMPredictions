import pandas as pd
import requests
import time
import math

# Path to the Excel file
file_path = "C:/Users/s_alizadehnia/Downloads/21sh.xlsx"

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

# Load the Excel file with all sheets
excel_data = pd.ExcelFile(file_path)

# Dictionary to store dataframes for each sheet after processing
updated_sheets = {}

# Process each sheet in the Excel file
for sheet_name in excel_data.sheet_names:
    print(sheet_name)
    data_df = pd.read_excel(file_path, sheet_name=sheet_name)

    # Check if 'latitude' and 'longitude' columns exist in the sheet
    if 'latitude' not in data_df.columns or 'longitude' not in data_df.columns:
        print(f"Skipping sheet {sheet_name} due to missing 'latitude' or 'longitude' columns.")
        continue  # Skip this sheet if columns are missing

    # Add the 'correct_altitude' column to the dataframe if it doesn't exist
    if 'correct_altitude' not in data_df.columns:
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

    # Store the updated dataframe in the dictionary
    updated_sheets[sheet_name] = data_df

# print(file_path[:-5])
# Save all updated sheets to a new Excel file
output_file_path = f"{file_path[:-5]}_alt_added.xlsx"
with pd.ExcelWriter(output_file_path, engine='xlsxwriter') as writer:
    for sheet_name, df in updated_sheets.items():
        df.to_excel(writer, sheet_name=sheet_name, index=False)

print(f"Updated file saved to {output_file_path}")
