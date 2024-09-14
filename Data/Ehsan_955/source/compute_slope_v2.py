import pandas as pd

# Define the file path for the Excel file
file_path = "C:/Users/s_alizadehnia/Downloads/21sh_alt_added.xlsx"
# Load the Excel file with all sheets
excel_data = pd.ExcelFile(file_path)

# Process each sheet in the Excel file
# for sheet_name in excel_data.sheet_names:
#   print(sheet_name)
#   if sheet_name == "Data":
#     # Read each sheet into a DataFrame
#     data_df = pd.read_excel(file_path, sheet_name=sheet_name)
#
#     # Initialize the 'slope' column to 0.0
#     data_df['slope'] = 0.0
#
#     # Calculate slope for each block of 100 rows
#     for i in range(0, len(data_df) - 100, 100):
#         start_altitude = data_df.at[i, 'correct_altitude']
#         end_altitude = data_df.at[i + 100, 'correct_altitude']
#         start_mileage = data_df.at[i, 'Cumulative_mileage']
#         end_mileage = data_df.at[i + 100, 'Cumulative_mileage']
#         if end_mileage - start_mileage != 0:
#             slope = (end_altitude - start_altitude) / (end_mileage - start_mileage) / 10
#             data_df.loc[i:i + 99, 'slope'] = slope
#
#     # Define CSV file path for each sheet
#     output_file_path = f'{file_path[:-5]}_slope_added.csv'
#
#     # Save the updated DataFrame to a CSV file
#     data_df.to_csv(output_file_path, index=False)
#
#     print(f"Updated file saved to {output_file_path}")
#
# print("All sheets have been processed and saved as separate CSV files.")


for sheet_name in excel_data.sheet_names:
    print(sheet_name)

    # Read each sheet into a DataFrame
    data_df = pd.read_excel(file_path, sheet_name=sheet_name)

    # Initialize the 'slope' column to 0.0
    data_df['slope'] = 0.0

    # Calculate slope for each block of 100 rows
    for i in range(0, len(data_df) - 100, 100):
        start_altitude = data_df.at[i, 'correct_altitude']
        end_altitude = data_df.at[i + 100, 'correct_altitude']
        start_mileage = data_df.at[i, 'Cumulative_mileage']
        end_mileage = data_df.at[i + 100, 'Cumulative_mileage']
        if end_mileage - start_mileage != 0:
            slope = (end_altitude - start_altitude) / (end_mileage - start_mileage) / 10
            data_df.loc[i:i + 99, 'slope'] = slope

    # Define CSV file path for each sheet
    output_file_path = f'{file_path[:-15]}_{sheet_name}_slope_added.csv'

    # Save the updated DataFrame to a CSV file
    data_df.to_csv(output_file_path, index=False)

    print(f"Updated file saved to {output_file_path}")

print("All sheets have been processed and saved as separate CSV files.")