import pandas as pd
import os

# Path to the folder containing CSV files
folder_path = "C:/Users/s_alizadehnia/Desktop/LSTMPredictions/Data/Ehsan_955/Qani_data/test_qani_data/valid_test"

# Process each CSV file in the specified folder
for filename in os.listdir(folder_path):
    if filename.endswith('.csv'):
        print(filename)
        file_path = os.path.join(folder_path, filename)
        print(f"Processing file: {filename}")

        # Read the CSV file
        data_df = pd.read_csv(file_path)
        data_df = data_df.sort_values(by='time')
        data_df['correct_altitude'] = data_df['correct_altitude'].fillna(method='bfill')

        # Sort the DataFrame by 'time' and 'trip'


        # Initialize the 'slope' column with 0.0
        data_df['slope'] = 0.0

        # Group by 'trip' and calculate slope for each group
        for trip, group in data_df.groupby('trip'):
            # Check if the group has at least 100 rows
            num_rows = len(group)
            for i in range(0, num_rows - 100, 100):
                start_altitude = group.iloc[i]['correct_altitude']
                end_altitude = group.iloc[i + 100]['correct_altitude']
                start_mileage = group.iloc[i]['Cumulative_mileage']
                end_mileage = group.iloc[i + 100]['Cumulative_mileage']

                if end_mileage - start_mileage != 0:
                    slope = (end_altitude - start_altitude) / (end_mileage - start_mileage) / 10
                    data_df.loc[group.index[i:i + 99], 'slope'] = slope

            # Calculate slope for remaining rows if less than 100 remain
            if num_rows % 100 != 0:
                last_block_start = num_rows - (num_rows % 100)
                if last_block_start > 0:
                    start_altitude = group.iloc[last_block_start - 1]['correct_altitude']
                    end_altitude = group.iloc[num_rows - 1]['correct_altitude']
                    start_mileage = group.iloc[last_block_start - 1]['Cumulative_mileage']
                    end_mileage = group.iloc[num_rows - 1]['Cumulative_mileage']

                    if end_mileage - start_mileage != 0:
                        slope = (end_altitude - start_altitude) / (end_mileage - start_mileage) / 10
                        data_df.loc[group.index[last_block_start - 1:num_rows - 1], 'slope'] = slope

        # Define output file path
        # output_file_path = os.path.join(folder_path, f"{filename[:-4]}_slope_added.csv")
        output_file_path = os.path.join(folder_path, filename)

        # Save the updated DataFrame to a new CSV file
        data_df.to_csv(output_file_path, index=False)

print("All files processed successfully.")
