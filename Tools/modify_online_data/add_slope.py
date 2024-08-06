import pandas as pd

slope_df = pd.read_excel('Slope.xlsx')

def percentage_to_float(percentage_str):
    return float(percentage_str)*100

slope_df['Slope_01'] = slope_df['Slope_01'].apply(percentage_to_float)
slope_df['Slope_02'] = slope_df['Slope_02'].apply(percentage_to_float)

for index, row in slope_df.iterrows():
    file_name = row['file_name'][-10:]  # Extract the last 10 characters as the real filename

    csv_df = pd.read_csv(file_name)

    half_length = len(csv_df) // 2
    new_column = [row['Slope_01']] * half_length + [row['Slope_02']] * (len(csv_df) - half_length)
    csv_df['slope'] = new_column

    csv_df.to_csv(file_name, index=False)

print("Columns added successfully to all CSV files.")
