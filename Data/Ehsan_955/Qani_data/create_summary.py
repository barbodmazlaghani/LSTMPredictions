import os
import pandas as pd

# Define function to calculate trip summaries
def summarize_trip_data(file_paths):
    summary_data = []

    # Process each CSV file
    for file_path in file_paths:
        df = pd.read_csv(file_path)

        # Trip-based calculations
        trip = df['trip'].unique()[0]  # Assuming one trip per file
        trip_start_time = df['timestamp'].min()
        trip_end_time = df['timestamp'].max()
        data_count = len(df)
        trip_time = (df['time'].max() - df['time'].min()) / 1000  # Convert ms to seconds
        trip_distance = df['Cumulative_mileage'].max() - df['Cumulative_mileage'].min()
        trip_idle_time = df[df['Vehicle_Speed'] == 0]['time'].count()  # Count idle times
        trip_speed_avg = df['Vehicle_Speed'].mean()
        trip_speed_std = df['Vehicle_Speed'].std()
        trip_speed_max = df['Vehicle_Speed'].max()
        trip_engine_speed_avg = df['Engine_speed'].mean()
        trip_acceleration_avg = df['angular_speed'].mean()
        trip_deceleration_avg = df[df['angular_speed'] < 0]['angular_speed'].mean()
        trip_accelerator_pedal_position_avg = df['Accelerator_pedal_position'].mean()
        trip_accelerator_pedal_position_std = df['Accelerator_pedal_position'].std()
        trip_coolant_temperature_avg = df['Coolant_temperature'].mean()
        trip_coolant_temperature_start = df['Coolant_temperature'].iloc[0]
        trip_fuel_consumption_lit = df['Trip_fuel_consumption'].max() - df['Trip_fuel_consumption'].min()
        trip_fuel_consumption_avg = df['Trip_fuel_consumption'].mean()

        # Append summary
        summary_data.append({
            'report_trip_number': trip,
            'trip_start_time': trip_start_time,
            'trip_end_time': trip_end_time,
            'data_count': data_count,
            'trip_time': trip_time,
            'trip_distance': trip_distance,
            'trip_idle_time': trip_idle_time,
            'trip_speed_avg': trip_speed_avg,
            'trip_speed_std': trip_speed_std,
            'trip_speed_max': trip_speed_max,
            'trip_engine_speed_avg': trip_engine_speed_avg,
            'trip_acceleration_avg': trip_acceleration_avg,
            'trip_deceleration_avg': trip_deceleration_avg,
            'trip_accelerator_pedal_position_avg': trip_accelerator_pedal_position_avg,
            'trip_accelerator_pedal_position_std': trip_accelerator_pedal_position_std,
            'trip_coolant_temperature_avg': trip_coolant_temperature_avg,
            'trip_coolant_temperature_start': trip_coolant_temperature_start,
            'trip_fuel_consumption_lit': trip_fuel_consumption_lit,
            'trip_fuel_consumption_avg': trip_fuel_consumption_avg
        })

    # Create DataFrame and save to CSV
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv('trip_summary.csv', index=False)
    return summary_df

# Example usage
folder_path = "C:/Users/s_alizadehnia/Desktop/LSTMPredictions/Data/Ehsan_955/Qani_data/qani"  # Replace with your folder path
file_paths = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith('.csv')]

# Generate trip summaries
trip_summary = summarize_trip_data(file_paths)

# Display the summary
print("Summary saved to trip_summary.csv")
