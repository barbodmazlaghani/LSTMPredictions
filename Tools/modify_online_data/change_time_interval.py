import pandas as pd

# Load the Excel file
file_path = input("FILE PATH : \n")
excel_data = pd.ExcelFile(file_path)

# Load the specific sheet
data_sheet = excel_data.parse('Data')

# Repeat each row 10 times
repeated_data = data_sheet.loc[data_sheet.index.repeat(10)].reset_index(drop=True)

# Calculate the average for every 8 rows
averaged_data = repeated_data.groupby(repeated_data.index // 8).mean()

# Save the modified data to a new Excel file
output_path = "modified.xlsx"
averaged_data.to_excel(output_path, index=False)

print("Data has been modified and saved to", output_path)
