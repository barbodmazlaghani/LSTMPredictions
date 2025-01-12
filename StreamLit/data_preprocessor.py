import os
import shutil
import pandas as pd
import numpy as np
import streamlit as st
from altitude_processor import process_altitudes
from slope_calculator import process_files_with_slope
from Augment1 import process_files_in_folder
from Augment2 import process_files_in_folder2
from Augment_coolant import process_files_in_folder3
from create_npz import augment


# Function to check if both time and fuel consumption are monotonic increasing
def is_monotonic_increasing_with_stability(time_series, fuel_series):
    """
    Ensure that time is strictly increasing, and fuel consumption either stays the same or increases.
    """
    return time_series.is_monotonic_increasing and all(fuel_series.diff().ge(0) | fuel_series.diff().eq(0))


def create_temp_folder():
    # Create a temporary folder (e.g., using a random or specific folder name)
    temp_folder = 'temp_folder'
    if not os.path.exists(temp_folder):
        os.makedirs(temp_folder)
    return temp_folder

def create_temp_folder2():
    # Create a temporary folder (e.g., using a random or specific folder name)
    temp_folder = 'temp_folder2'
    if not os.path.exists(temp_folder):
        os.makedirs(temp_folder)
    return temp_folder

def delete_temp_folder(folder_path):
    # Delete the folder and all its contents after processing is done
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)

# Function to pad or truncate sequences
def pad_or_truncate(sequence, length):
    if len(sequence) > length:
        return sequence[:length]
    elif len(sequence) < length:
        return np.pad(sequence, ((0, length - len(sequence)), (0, 0)), mode='constant')
    else:
        return sequence



def check_len_rows(folder_path, min_rows=100):
    """
    Checks the number of rows in each CSV file within a folder.
    Deletes files with rows less than the specified minimum.

    Parameters:
    - folder_path (str): Path to the folder containing CSV files.
    - min_rows (int): Minimum number of rows a file must have to avoid deletion. Default is 100.

    Returns:
    - None
    """
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.csv'):  # Check if the file is a CSV
            file_path = os.path.join(folder_path, file_name)
            try:
                # Read the CSV file into a DataFrame
                df = pd.read_csv(file_path)

                # Check the number of rows in the DataFrame
                if len(df) < min_rows:
                    print(f"Deleting {file_name} (rows: {len(df)})")
                    os.remove(file_path)  # Delete the file
                else:
                    print(f"Keeping {file_name} (rows: {len(df)})")
            except Exception as e:
                print(f"Error processing {file_name}: {e}")

# Example usage
# folder_path = "path_to_your_folder"
# check_len_rows(folder_path, min_rows=100)


def preprocess_and_save(df, selected_columns):
    """
    Preprocesses the data based on selected columns and processes trip data if the trip_column is available.

    Parameters:
    - df: DataFrame containing the original data.
    - selected_columns: Dictionary of selected columns from the user.
    """
    # Create a temporary folder to save the processed files
    temp_folder = create_temp_folder()
    
    # Check if trip column exists
    if selected_columns["trip_column"] is not None:
        # Grouping by the selected trip column
        grouped = df.groupby(selected_columns["trip_column"])

        # Iterate through each trip group
        for trip, group in grouped:

            # Check if both 'time' and selected_columns["fuel_column"] are monotonic increasing
            time_column = selected_columns["time_column"]
            fuel_column = selected_columns["fuel_column"]

            if not is_monotonic_increasing_with_stability(group[time_column], group[fuel_column]):
                # Split the group into two or more based on where time or fuel consumption decreases
                split_groups = []
                current_group = []
                last_time = None
                last_fuel = None

                for index, row in group.iterrows():
                    # Ensure time is strictly increasing and fuel consumption doesn't decrease
                    if last_time is None or (row[time_column] >= last_time and (row[fuel_column] >= last_fuel or row[fuel_column] == last_fuel)):
                        current_group.append(row)
                    else:
                        # When time or fuel consumption decreases, store the current group and start a new one
                        split_groups.append(pd.DataFrame(current_group))
                        current_group = [row]
                    last_time = row[time_column]
                    last_fuel = row[fuel_column]

                # Add the last group
                split_groups.append(pd.DataFrame(current_group))

                # Save each split group as a new CSV file in the temp folder
                for i, split_group in enumerate(split_groups):
                    filename = f"{temp_folder}/trip_{trip}_part_{i+1}.csv"
                    split_group.to_csv(filename, index=False)

            else:
                # If both time and fuel consumption are monotonic, save the group as a CSV file
                filename = f"{temp_folder}/trip_{trip}.csv"
                group.to_csv(filename, index=False)
    
        else:
            df[selected_columns["engine_speed"]] = pd.to_numeric(df[selected_columns["engine_speed"]], errors='coerce')
            df[selected_columns["fuel_column"]] = pd.to_numeric(df[selected_columns["fuel_column"]], errors='coerce')
            df[selected_columns["time_column"]] = pd.to_numeric(df[selected_columns["time_column"]], errors='coerce')
            df[selected_columns["voltage"]] = pd.to_numeric(df[selected_columns["voltage"]], errors='coerce')
            # If no trip_column, identify trips based on Engine_speed and Battery_voltage
            trip_number = 1
            trip_data = []
            current_trip = []

            for index, row in df.iterrows():
                engine_speed = row[selected_columns["engine_speed"]]
                battery_voltage = row[selected_columns["voltage"]]

                if engine_speed > 0 and battery_voltage > 0:  # Start or continue a trip
                    current_trip.append(row)
                else:  # End the current trip
                    if current_trip:
                        trip_data.append(current_trip)
                        current_trip = []

            # Save the last trip if exists
            if current_trip:
                trip_data.append(current_trip)

            # Process each identified trip
            for idx, trip in enumerate(trip_data):
                trip_df = pd.DataFrame(trip)

                # Check monotonicity
                if not is_monotonic_increasing_with_stability(trip_df[selected_columns["time_column"]], trip_df[selected_columns["fuel_column"]]):
                    # Split based on monotonicity
                    split_groups = []
                    current_group = []
                    last_time = None
                    last_fuel = None

                    for index, row in trip_df.iterrows():
                        if last_time is None or (row[selected_columns["time_column"]] >= last_time and (row[selected_columns["fuel_column"]] >= last_fuel or row[selected_columns["fuel_column"]] == last_fuel)):
                            current_group.append(row)
                        else:
                            split_groups.append(pd.DataFrame(current_group))
                            current_group = [row]
                        last_time = row[selected_columns["time_column"]]
                        last_fuel = row[selected_columns["fuel_column"]]

                    split_groups.append(pd.DataFrame(current_group))

                    # Save each split group
                    for i, split_group in enumerate(split_groups):
                        filename = f"{temp_folder}/part_{idx+1}_{i+1}"
                        split_group.to_csv(filename, index=False)
                        print(f"Saved {filename}")
                else:
                    # Save monotonic trip
                    filename = f"{temp_folder}/trip_{trip_number}"
                    trip_df.to_csv(filename, index=False)
                    print(f"Saved {filename}")
                    trip_number += 1


    check_len_rows(temp_folder, min_rows=50)


    process_altitudes(temp_folder , selected_columns)

    process_files_with_slope(temp_folder , selected_columns)

    temp_folder2 = create_temp_folder2()

    # Define source and target folders, and processing parameters
    source_folder = temp_folder
    target_folder = temp_folder2
    num_slices = 2
    target_rows = 10000000000000

    # Call the processing function
    process_files_in_folder(source_folder, target_folder, num_slices, target_rows ,selected_columns)
    process_files_in_folder2(source_folder, target_folder, num_slices, target_rows , selected_columns)
    process_files_in_folder3(source_folder, target_folder, num_slices, target_rows , selected_columns)
    
    X_augmented, y_augmented , SEQUENCE_LENGTH = augment(temp_folder2,selected_columns )

    X_original = []
    y_original = []
    # Apply padding/truncating to ensure consistent sequence length
    X_original = [pad_or_truncate(x, SEQUENCE_LENGTH) for x in X_original]
    y_original = [pad_or_truncate(y.reshape(-1, 1), SEQUENCE_LENGTH) for y in y_original]
    X_augmented = [pad_or_truncate(x, SEQUENCE_LENGTH) for x in X_augmented]
    y_augmented = [pad_or_truncate(y.reshape(-1, 1), SEQUENCE_LENGTH) for y in y_augmented]


    X_original = np.array(X_original)
    y_original = np.array(y_original)
    X_augmented = np.array(X_augmented)
    y_augmented = np.array(y_augmented)

    delete_temp_folder(temp_folder)
    delete_temp_folder(temp_folder2)

    save_path = 'train_file.npz'
    # Saving to npz file
    np.savez(save_path, X_original=X_original, y_original=y_original, X_augmented=X_augmented, y_augmented=y_augmented)
    # نمایش پیام مسیر ذخیره‌شدن در استریم‌لیت

    st.write(f"فایل با موفقیت در مسیر زیر ذخیره شد: {os.path.abspath(save_path)}")

    # امکان دانلود فایل برای کاربر
    with open(save_path, 'rb') as file:
        st.download_button(label="دانلود فایل npz", data=file, file_name=save_path, mime="application/octet-stream")
        