import pandas as pd


def split_excel_file(input_file, output_file_prefix, rows_per_file):
    df = pd.read_excel(input_file)

    total_rows = len(df)
    num_files = (total_rows // rows_per_file) + (1 if total_rows % rows_per_file != 0 else 0)

    for i in range(num_files):
        start_row = i * rows_per_file
        end_row = start_row + rows_per_file

        subset_df = df.iloc[start_row:end_row]

        output_file = f"{output_file_prefix}_{i + 1}.csv"

        subset_df.to_csv(output_file, index=False)

    print(f"Split into {num_files} files.")



input_file = "C://Users//bamir//Desktop//Repos//LSTMPredictions//Data//Rajab_641//trip.xlsx"
output_file_prefix = "output_file"
rows_per_file = 610

split_excel_file(input_file, output_file_prefix, rows_per_file)
