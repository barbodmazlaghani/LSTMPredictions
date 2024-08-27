import os
import pandas as pd

# Get directory from the user
directory = input("Please enter the directory containing Excel files: ")

# Check if the directory exists
if not os.path.isdir(directory):
    print("The provided directory does not exist.")
else:
    # List all files in the directory
    for file_name in os.listdir(directory):
        if file_name.endswith('.xlsx') or file_name.endswith('.xls'):
            # Construct full file path
            file_path = os.path.join(directory, file_name)

            # Load Excel file
            excel_data = pd.ExcelFile(file_path)

            # Loop through each sheet in the Excel file
            for sheet_name in excel_data.sheet_names:
                # Read the sheet into a DataFrame
                df = pd.read_excel(file_path, sheet_name=sheet_name)

                # Define CSV file path
                csv_file_name = f"{os.path.splitext(file_name)[0]}_{sheet_name}.csv"
                csv_file_path = os.path.join(directory, csv_file_name)

                # Save DataFrame to CSV
                df.to_csv(csv_file_path, index=False)

                print(f"Saved {sheet_name} to {csv_file_path}")

    print("All sheets have been saved as CSV files.")
