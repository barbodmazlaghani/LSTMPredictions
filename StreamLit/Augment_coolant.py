import os
import pandas as pd
import itertools
import random  # For random sampling
from tqdm import tqdm  # For progress tracking

import streamlit as st

def adjust_sequence(prev_value, slice_values):
    adjusted_values = [prev_value]
    for i in range(len(slice_values)):
        delta = slice_values[i] - slice_values[i - 1] if i > 0 else 0
        adjusted_values.append(adjusted_values[-1] + delta)
    return adjusted_values[1:]

def process_combination(slices , selected_columns):
    processed_data = pd.DataFrame()
    num_slices = len(slices)

    for i in range(num_slices):
        if i > 0:
            if processed_data.empty:
                continue

            coolant_temp_prev = processed_data[selected_columns["coolant_column"]].iloc[-1]
            coolant_temp_current = slices[i][selected_columns["coolant_column"]].iloc[0]

            if abs(coolant_temp_current - coolant_temp_prev) > 10:
                continue

            for column in [selected_columns["time_column"], selected_columns["mileage_column"], selected_columns["fuel_column"]]:
                prev_value = processed_data[column].iloc[-1]
                slices[i].loc[:, column] = adjust_sequence(prev_value, list(slices[i][column]))

            slices[i] = slices[i].iloc[1:]

        processed_data = pd.concat([processed_data, slices[i]], ignore_index=True)

    return processed_data


import os
import pandas as pd
import itertools
import random
import streamlit as st

def process_files_in_folder3(source_folder, target_folder, num_slices, target_rows , selected_columns):
    os.makedirs(target_folder, exist_ok=True)

    # Streamlit progress bars
    total_files = len([f for f in os.listdir(source_folder) if f.endswith('.csv')])
    file_progress = st.progress(0)
    combination_progress = st.progress(0)
    status_text = st.empty()

    # Process files in the source folder
    driver_files = {}
    file_count = 0

    for idx, filename in enumerate(os.listdir(source_folder)):
        if filename.endswith('.csv'):
            file_path = os.path.join(source_folder, filename)
            df = pd.read_csv(file_path)

            # Rename columns
            # df.rename(columns={
            #     'Engine_speed': 'Engine speed',
            #     'time': 'Time',
            #     'Vehicle_Speed': 'Speed',
            #     'Trip_fuel_consumption': 'Trip fuel consumption',
            #     'Throttle_position': 'Throttle position',
            #     'Accelerator_pedal_position': 'Accelerator pedal position',
            #     'Cumulative_mileage': 'Cumulative mileage',
            #     'Coolant_temp': selected_columns["coolant_column"]
            # }, inplace=True)

            slice_size = len(df) // num_slices
            driver_name = filename[:-4]

            mileage_slices = [df.iloc[i * slice_size:(i + 1) * slice_size] for i in range(num_slices)]
            mileage_slices[-1] = pd.concat([mileage_slices[-1], df.iloc[(num_slices * slice_size):]])

            driver_files[driver_name] = mileage_slices

            # Update file progress
            file_progress.progress(min((idx + 1) / total_files, 1.0))


    # Filter drivers with slices starting where Coolant_temperature < 30
    filtered_drivers = {}
    for driver, slices in driver_files.items():
        filtered_slices = [s for s in slices if s[selected_columns["coolant_column"]].iloc[0] < 30]
        if filtered_slices:
            filtered_drivers[driver] = filtered_slices

    # Generate combinations with the first slice starting at Coolant_temperature < 30
    driver_combinations = list(itertools.product(filtered_drivers.keys(), repeat=num_slices))
    random.shuffle(driver_combinations)

    total_rows = 0
    file_count = 0

    for idx, combination in enumerate(driver_combinations):
        # Ensure the first slice comes from filtered drivers
        first_driver = combination[0]
        slices = [random.choice(filtered_drivers[first_driver])]
        slices += [random.choice(driver_files[driver]) for driver in combination[1:]]

        combined_data = process_combination(slices , selected_columns)

        if len(combined_data) >= 1:
            output_file = os.path.join(target_folder, f"{'_'.join([f'{driver}(slice_{i + 1})' for i, driver in enumerate(combination)])}_H.csv")
            combined_data.to_csv(output_file, index=False)
            total_rows += len(combined_data)
            file_count += 1

            # Update status text
            status_text.text(f"Processed {file_count} files with {total_rows} rows.")

        # Update combination progress
        combination_progress.progress((idx + 1) / len(driver_combinations))

        if total_rows >= target_rows:
            status_text.text(f"Reached target of {target_rows} rows. Stopping further processing.")
            break

    st.write(f"Processed and saved {file_count} files with a total of {total_rows} rows.")
