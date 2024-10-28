import pandas as pd

def separate_data_by_engine_speed_zero_borders(file_path, column_name='Engine_speed'):
    df = pd.read_csv(file_path)

    zero_speed_indices = df.index[df[column_name] == 0].tolist()
    borders = [0]

    for idx in zero_speed_indices:
        if idx - 1 not in borders:
            borders.append(idx)

    borders.append(len(df))

    file_paths = []
    for i in range(len(borders) - 1):
        start_idx, end_idx = borders[i], borders[i + 1]
        segment = df.iloc[start_idx:end_idx]
        if not segment.empty:
            file_name = f'segment_{start_idx}_{end_idx}.csv'
            segment.to_csv(file_name, index=False)
            file_paths.append(file_name)

    return file_paths

file_path = 'E:\علم داده\RajabTrips\Device_7193_Data_01_01_2024, 13_49_00_to_01_14_2024, 13_49_00.csv'


separated_files = separate_data_by_engine_speed_zero_borders(file_path)
print(separated_files)
