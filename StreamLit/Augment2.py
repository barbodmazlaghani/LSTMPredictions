import os
import pandas as pd
import itertools
from tqdm import tqdm
import streamlit as st

def adjust_sequence(prev_value, slice_values):
    """
    Adjust a sequence based on the previous value and differences in the slice.
    """
    adjusted_values = [prev_value]
    for i in range(len(slice_values)):
        delta = slice_values[i] - slice_values[i - 1] if i > 0 else 0
        adjusted_values.append(adjusted_values[-1] + delta)
    return adjusted_values[1:]


def process_combination(slices , selected_columns):
    """
    Process a combination of data slices and return the combined DataFrame.
    """
    processed_data = pd.DataFrame()
    num_slices = len(slices)

    for i in range(num_slices):
        if i > 0:
            # Check Coolant_temperature condition before adjusting sequence
            coolant_temp_prev = processed_data[selected_columns["coolant_column"]].iloc[-1]
            coolant_temp_current = slices[i][selected_columns["coolant_column"]].iloc[0]

            # Skip slice if Coolant_temperature difference is greater than 10°C
            if abs(coolant_temp_current - coolant_temp_prev) > 10:
                continue

            # Adjust sequences for specified columns
            for column in [selected_columns["time_column"], selected_columns["mileage_column"], selected_columns["fuel_column"]]:
                if not processed_data.empty:
                    prev_value = processed_data[column].iloc[-1]
                else:
                    prev_value = slices[i][column].iloc[0]  # Fallback in case of empty processed_data

                slices[i].loc[:, column] = adjust_sequence(prev_value, list(slices[i][column]))

            slices[i] = slices[i].iloc[1:]  # Skip the first row of the current slice (already adjusted)

        processed_data = pd.concat([processed_data, slices[i]], ignore_index=True)

    return processed_data


def process_files_in_folder2(source_folder, target_folder, num_slices, target_rows , selected_columns):
    os.makedirs(target_folder, exist_ok=True)

    driver_files = {}
    progress_bar = st.progress(0)
    total_files = len(os.listdir(source_folder))
    file_counter = 0

    for filename in os.listdir(source_folder):
        if filename.endswith('.csv'):
            file_path = os.path.join(source_folder, filename)
            df = pd.read_csv(file_path)

            slice_size = len(df) // num_slices
            driver_name = filename[:-4]

            mileage_slices = [df.iloc[i * slice_size:(i + 1) * slice_size] for i in range(num_slices)]
            mileage_slices[-1] = pd.concat([mileage_slices[-1], df.iloc[(num_slices * slice_size):]])

            driver_files[driver_name] = mileage_slices

        file_counter += 1
        progress_bar.progress(file_counter / total_files)

    file_count = 0
    total_rows = 0
    combination_progress = st.progress(0)
    total_combinations = sum(len(list(itertools.product(range(len(slices)), repeat=num_slices))) for slices in driver_files.values())
    processed_combinations = 0

    for driver, slices in driver_files.items():
        slice_indices = range(len(slices))
        slice_combinations = list(itertools.product(slice_indices, repeat=num_slices))

        for combination in slice_combinations:
            selected_slices = [slices[index] for index in combination]
            combined_data = process_combination(selected_slices , selected_columns)

            if len(combined_data) > 1:
                output_file = os.path.join(
                    target_folder,
                    f"{driver}_" + "_".join([f"slice_{index + 1}" for index in combination]) + ".csv"
                )
                combined_data.to_csv(output_file, index=False)
                total_rows += len(combined_data)
                file_count += 1

            processed_combinations += 1
            combination_progress.progress(processed_combinations / total_combinations)

            if total_rows >= target_rows:
                st.write(f"Reached target of {target_rows} rows. Stopping further processing.")
                return
    st.write(f"Processed {file_count} files with {total_rows} rows.")