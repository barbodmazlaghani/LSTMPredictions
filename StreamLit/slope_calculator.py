import pandas as pd
import os
import streamlit as st

def process_files_with_slope(folder_path, selected_columns):
    """
    Processes all CSV files in a specified folder to calculate slopes for a single trip in each file.
    Updates the files with a new 'slope' column and integrates with Streamlit for progress tracking.

    Parameters:
    - folder_path (str): Path to the folder containing CSV files.
    - selected_columns (dict): Dictionary specifying column names like time, altitude, and mileage.

    Returns:
    - None
    """
    # List all CSV files in the folder
    csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
    total_files = len(csv_files)

    # Add a progress bar to the Streamlit app
    progress_bar = st.progress(0)
    status_text = st.empty()  # To display file processing status

    for idx, filename in enumerate(csv_files):
        file_path = os.path.join(folder_path, filename)
        status_text.text(f"Processing file {idx + 1}/{total_files}: {filename}")

        # Read the CSV file
        data_df = pd.read_csv(file_path)
        # data_df = data_df.sort_values(by=selected_columns["time_column"])

        # Fill missing 'correct_altitude' values using backward fill
        data_df['correct_altitude'] = data_df['correct_altitude'].fillna(method='bfill')

        # Initialize the 'slope' column with 0.0
        data_df['slope'] = 0.0

        # Process the file row-wise in blocks of 100
        num_rows = len(data_df)
        for i in range(0, num_rows - 100, 100):
            start_altitude = data_df.iloc[i]['correct_altitude']
            end_altitude = data_df.iloc[i + 100]['correct_altitude']
            start_mileage = data_df.iloc[i][selected_columns["mileage_column"]]
            end_mileage = data_df.iloc[i + 100][selected_columns["mileage_column"]]

            if end_mileage - start_mileage != 0:
                slope = (end_altitude - start_altitude) / (end_mileage - start_mileage) / 10
                data_df.loc[i:i + 99, 'slope'] = slope

        # Calculate slope for remaining rows if less than 100 remain
        if num_rows % 100 != 0:
            last_block_start = num_rows - (num_rows % 100)
            if last_block_start > 0:
                start_altitude = data_df.iloc[last_block_start - 1]['correct_altitude']
                end_altitude = data_df.iloc[num_rows - 1]['correct_altitude']
                start_mileage = data_df.iloc[last_block_start - 1][selected_columns["mileage_column"]]
                end_mileage = data_df.iloc[num_rows - 1][selected_columns["mileage_column"]]

                if end_mileage - start_mileage != 0:
                    slope = (end_altitude - start_altitude) / (end_mileage - start_mileage) / 10
                    data_df.loc[last_block_start - 1:num_rows - 1, 'slope'] = slope

        # Save the updated DataFrame back to the same file
        output_file_path = os.path.join(folder_path, filename)
        data_df.to_csv(output_file_path, index=False)

        # Update the progress bar
        progress_bar.progress((idx + 1) / total_files)

    # Completion message
    status_text.text("All files added slope successfully!")
