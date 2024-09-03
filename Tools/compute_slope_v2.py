import pandas as pd

# Define the file path for the Excel file
file_path = 'C:/Users/s_alizadehnia/Desktop/LSTMPredictions/Data/Rajab_641/generated/trip_8_3(date_8_13).csv'
output_file_path = f'8_3_slope_added.csv'
# Load the Excel file with all sheets
# excel_data = pd.ExcelFile(file_path)




data_df = pd.read_csv(file_path)

# Initialize the 'slope' column to 0.0
data_df['slope'] = 0.0

# Calculate slope for each block of 100 rows
for i in range(0, len(data_df) - 100, 100):
    start_altitude = data_df.at[i, 'altitude']
    end_altitude = data_df.at[i + 100, 'altitude']
    start_mileage = data_df.at[i, 'Cumulative_mileage']
    end_mileage = data_df.at[i + 100, 'Cumulative_mileage']
    if end_mileage - start_mileage != 0:
        slope = (end_altitude - start_altitude) / (end_mileage - start_mileage) / 10
        data_df.loc[i:i + 99, 'slope'] = slope

# Define CSV file path for each sheet


# Save the updated DataFrame to a CSV file
data_df.to_csv(output_file_path, index=False)

print(f"Updated file saved to {output_file_path}")

print("All sheets have been processed and saved as separate CSV files.")
