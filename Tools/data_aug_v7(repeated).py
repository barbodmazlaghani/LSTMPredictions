import pandas as pd
import os
import itertools

def adjust_sequence(prev_value, slice_values):
    adjusted_values = [prev_value]
    for i in range(len(slice_values)):
        delta = slice_values[i] - slice_values[i - 1] if i > 0 else 0
        adjusted_values.append(adjusted_values[-1] + delta)
    return adjusted_values[1:]

def process_combination(slices):
    processed_data = pd.DataFrame()
    num_slices = len(slices)
    for i in range(num_slices):
        if i > 0:
            for column in ['Time', 'Cumulative mileage', 'Trip fuel consumption']:
                prev_value = processed_data[column].iloc[-1]
                slices[i][column] = adjust_sequence(prev_value, list(slices[i][column]))
            slices[i] = slices[i].iloc[1:]
        processed_data = pd.concat([processed_data, slices[i]])
    return processed_data

num_slices = 3

driver_data = pd.read_excel('E:\\علم داده\\Project_01\\drivers_refined\\driver-data.xlsx')
source_folder = 'E:\\علم داده\\Project_01\\drivers_refined\\shahrivar\\11_shahrivar\\cluster_5'
target_folder = 'E:\\علم داده\\Project_01\\drivers_refined\\shahrivar\\11_shahrivar\\data_aug(3_slices_with_repeated)_cluster_5'

os.makedirs(target_folder, exist_ok=True)

driver_files = {}
for filename in os.listdir(source_folder):
    if filename.endswith('.csv'):
        file_path = os.path.join(source_folder, filename)
        df = pd.read_csv(file_path)
        # comment this for offline data loggers
        df.rename(columns={'Engine_speed': 'Engine speed',
                           'time': 'Time',
                           'Vehicle_Speed': 'Speed',
                           'Trip_fuel_consumption': 'Trip fuel consumption',
                           'Throttle_position': 'Throttle position',
                           'Accelerator_pedal_position': 'Accelerator pedal position',
                           'Cumulative_mileage': 'Cumulative mileage'
                           }, inplace=True)

        min_mileage = df['Cumulative mileage'].min()
        max_mileage = df['Cumulative mileage'].max()
        slice_size = (max_mileage - min_mileage) / num_slices

        # driver_name = driver_data.loc[driver_data['fileName'].str.contains(filename[:-4], na=False), 'driver_name_english'].values[0]
        driver_name = filename[:-4]
        mileage_slices = [df[(df['Cumulative mileage'] >= min_mileage + i * slice_size) &
                             (df['Cumulative mileage'] < min_mileage + (i + 1) * slice_size)]
                          for i in range(num_slices)]

        mileage_slices[-1] = pd.concat([mileage_slices[-1], df[df['Cumulative mileage'] == max_mileage]])
        driver_files[driver_name] = mileage_slices

driver_combinations = list(itertools.product(driver_files.keys(), repeat=num_slices))

file_count = 0
for combination in driver_combinations:
    slices = [driver_files[driver][slice_index] for driver, slice_index in zip(combination, range(num_slices))]
    combined_data = process_combination(slices)
    output_file = os.path.join(target_folder,
                               f"{'_'.join([f'{driver}(slice_{slice_index + 1})' for driver, slice_index in zip(combination, range(num_slices))])}.csv")
    combined_data.to_csv(output_file, index=False)
    file_count += 1
