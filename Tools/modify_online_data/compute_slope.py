import pandas as pd

file_path = 'updated_file_with_altitudes.xlsx'
data_df = pd.read_excel(file_path)

data_df['slope'] = 0.0

for i in range(0, len(data_df) - 100, 100):
    start_altitude = data_df.at[i, 'correct_altitude']
    end_altitude = data_df.at[i + 100, 'correct_altitude']
    start_mileage = data_df.at[i, 'Cumulative_mileage']
    end_mileage = data_df.at[i + 100, 'Cumulative_mileage']
    if end_mileage - start_mileage != 0:
        slope = (end_altitude - start_altitude) / (end_mileage - start_mileage) / 10
        data_df.loc[i:i+99, 'slope'] = slope

output_file_path = 'updated_file_with_altitudes_and_slope.xlsx'
data_df.to_excel(output_file_path, index=False)

print(f"Updated file saved to {output_file_path}")