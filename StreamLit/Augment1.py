import os
import pandas as pd
import itertools
import random  # For random sampling
from tqdm import tqdm  # For progress tracking
import streamlit as st


def adjust_sequence(prev_value, slice_values):
    """
    Adjusts a sequence of values by calculating deltas from the previous value.

    Parameters:
    - prev_value (float): The previous value to start adjustment from.
    - slice_values (list): The list of values to adjust.

    Returns:
    - list: Adjusted sequence of values.
    """
    adjusted_values = [prev_value]
    for i in range(len(slice_values)):
        delta = slice_values[i] - slice_values[i - 1] if i > 0 else 0
        adjusted_values.append(adjusted_values[-1] + delta)
    return adjusted_values[1:]

def process_combination(slices , selected_columns):
    """
    Processes a combination of slices by adjusting sequences and merging them.

    Parameters:
    - slices (list of pd.DataFrame): List of slices to process.

    Returns:
    - pd.DataFrame: Processed DataFrame combining all slices.
    """
    processed_data = pd.DataFrame()
    num_slices = len(slices)

    for i in range(num_slices):
        if i > 0:
            # Skip if processed_data is empty or temperature difference > 10°C
            if processed_data.empty:
                continue

            coolant_temp_prev = processed_data[selected_columns["coolant_column"]].iloc[-1]
            coolant_temp_current = slices[i][selected_columns["coolant_column"]].iloc[0]

            if abs(coolant_temp_current - coolant_temp_prev) > 10:
                continue

            # Adjust sequences for specified columns
            for column in [selected_columns["time_column"], selected_columns["mileage_column"], selected_columns["fuel_column"]]:
                prev_value = processed_data[column].iloc[-1] if not processed_data.empty else slices[i][column].iloc[0]
                slices[i].loc[:, column] = adjust_sequence(prev_value, list(slices[i][column]))

            slices[i] = slices[i].iloc[1:]  # Skip the first row

        processed_data = pd.concat([processed_data, slices[i]], ignore_index=True)

    return processed_data

def process_files_in_folder(source_folder, target_folder, num_slices, target_rows , selected_columns):
    """
    Processes files in a source folder to generate augmented datasets.

    Parameters:
    - source_folder (str): Path to the source folder containing CSV files.
    - target_folder (str): Path to save processed files.
    - num_slices (int): Number of slices for each file.
    - target_rows (int): Total number of rows to generate.

    Returns:
    - None
    """
    os.makedirs(target_folder, exist_ok=True)
    driver_files = {}

    # Load files and slice data
    st.write("Loading files...")
    file_list = [f for f in os.listdir(source_folder) if f.endswith('.csv')]
    total_files = len(file_list)
    load_progress = st.progress(0)

    for idx, filename in enumerate(file_list):
        file_path = os.path.join(source_folder, filename)
        df = pd.read_csv(file_path)

        # df.rename(columns={
        #     'Engine_speed': 'Engine speed',
        #     'time': 'Time',
        #     'Vehicle_Speed': 'Speed',
        #     'Trip_fuel_consumption': 'Trip fuel consumption',
        #     'Throttle_position': 'Throttle position',
        #     'Accelerator_pedal_position': 'Accelerator pedal position',
        #     'Cumulative_mileage': 'Cumulative mileage',
        #     'Coolant_temp': 'Coolant_temperature'
        # }, inplace=True)

        slice_size = len(df) // num_slices
        driver_name = filename[:-4]
        mileage_slices = [df.iloc[i * slice_size:(i + 1) * slice_size] for i in range(num_slices)]
        mileage_slices[-1] = pd.concat([mileage_slices[-1], df.iloc[(num_slices * slice_size):]])
        driver_files[driver_name] = mileage_slices

        load_progress.progress((idx + 1) / total_files)

    st.write("Files loaded successfully.")

    driver_combinations = list(itertools.product(driver_files.keys(), repeat=num_slices))
    random.shuffle(driver_combinations)

    total_rows = 0
    combination_progress = st.progress(0)
    combination_status = st.empty()

    for idx, combination in enumerate(driver_combinations):
        slices = [random.choice(driver_files[driver]) for driver in combination]
        combined_data = process_combination(slices , selected_columns)

        if len(combined_data) >= 1:
            output_file = os.path.join(target_folder, f"{'_'.join([f'{driver}(slice_{i + 1})' for i, driver in enumerate(combination)])}.csv")
            combined_data.to_csv(output_file, index=False)
            total_rows += len(combined_data)

        combination_progress.progress((idx + 1) / len(driver_combinations))
        combination_status.write(f"Processed combinations: {idx + 1}/{len(driver_combinations)}")

        if total_rows >= target_rows:
            st.write(f"Reached target of {target_rows} rows. Stopping further processing.")
            break

    st.write(f"Processing completed. Total rows: {total_rows}")

# # Example usage
# source_folder = '/content/513'
# target_folder = '/content/513_augmented'
# num_slices = 3
# target_rows = 50000
# process_files_in_folder(source_folder, target_folder, num_slices, target_rows)
