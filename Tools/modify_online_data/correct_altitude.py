import pandas as pd
import requests
import time

file_path = 'Device_DNUM_Data_07_21_2024, 15_54_00_to_07_28_2024, 15_54_00_org.xlsx'
data_df = pd.read_excel(file_path, sheet_name='Data')

# count = 1
def get_altitude(lat, lon, retries=10):
    url = f"https://api.open-elevation.com/api/v1/lookup?locations={lat},{lon}"
    for attempt in range(retries):
        try:
            print("TRY")
            response = requests.get(url)
            if response.status_code == 200:
                elevation_data = response.json()
                if elevation_data['results']:
                    # print("Request : ",count)
                    # count += 1
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

# Get the correct altitudes for each block of 100 rows
for i in range(0, len(data_df), 100):
    altitude = get_altitude_for_block(data_df, i)
    data_df.loc[i:i+99, 'correct_altitude'] = altitude

# Save the updated DataFrame to a new Excel file
output_file_path = 'updated_file_with_altitudes.xlsx'
data_df.to_excel(output_file_path, index=False)

print(f"Updated file saved to {output_file_path}")
