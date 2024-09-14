import pandas as pd
import requests
import time

# Path to the Excel file
file_path = 'Data/Ehsan_955/source/Car_Dena_93 h 955 - ir 44_Data_08_30_2024, 11_19_00_to_09_01_2024, 23_20_00.xlsx'

# Function to get altitude
def get_altitude(lat, lon, retries=10):
    url = f"https://api.open-elevation.com/api/v1/lookup?locations={lat},{lon}"
    for attempt in range(retries):
        try:
            print("TRY")
            response = requests.get(url)
            if response.status_code == 200:
                elevation_data = response.json()
                if elevation_data['results']:
                    print(elevation_data)
                    return elevation_data['results'][0]['elevation']
        except requests.exceptions.RequestException as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            time.sleep(2)  # Wait for 2 seconds before retrying
    return None

# Helper function to get altitude for a block of 100 rows
def get_altitude_for_block(df, start_index):
    for i in range(start_index, min(start_index + 100, len(df))):
        if df.at[i, 'latitude'] != 0 and df.at[i, 'longitude'] != 0:
            altitude = get_altitude(df.at[i, 'latitude'], df.at[i, 'longitude'])
            if altitude is not None:
                return altitude
    return None

# Load the Excel file with all sheets
excel_data = pd.ExcelFile(file_path)

# Dictionary to store dataframes for each sheet after processing
updated_sheets = {}

# Process each sheet in the Excel file
for sheet_name in excel_data.sheet_names:
    data_df = pd.read_excel(file_path, sheet_name=sheet_name)

    # Add the 'correct_altitude' column to the dataframe if it doesn't exist
    if 'correct_altitude' not in data_df.columns:
        data_df['correct_altitude'] = None

    # Get the correct altitudes for each block of 100 rows
    for i in range(0, len(data_df), 100):
        altitude = get_altitude_for_block(data_df, i)
        data_df.loc[i:i + 99, 'correct_altitude'] = altitude

    # Store the updated dataframe in the dictionary
    updated_sheets[sheet_name] = data_df

# Save all updated sheets to a new Excel file
output_file_path = 'ehsan/Car_Dena_93 h 955 - ir 44_Data_08_21_2024, 14_00_00_to_08_21_2024, 16_20_00_Siamak_Kan_SaeedAbad_Bargasht_Wedensday_alt_correct.xlsx'
with pd.ExcelWriter(output_file_path, engine='xlsxwriter') as writer:
    for sheet_name, df in updated_sheets.items():
        df.to_excel(writer, sheet_name=sheet_name, index=False)

print(f"Updated file saved to {output_file_path}")
