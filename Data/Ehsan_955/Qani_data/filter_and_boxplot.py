import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Configure plot aesthetics (optional)
sns.set(style="whitegrid")


def identify_valid_trips(data_df, trip_column='trip', time_column='time', min_interval=400, max_interval=2000,
                         min_rows=1000):
    """
    Identifies valid trips based on time interval conditions and minimum row count.

    Parameters:
        data_df (pd.DataFrame): DataFrame containing trip and time data.
        trip_column (str): Name of the column representing trip identifiers.
        time_column (str): Name of the column representing time entries.
        min_interval (int): Minimum allowed time interval between consecutive rows.
        max_interval (int): Maximum allowed time interval between consecutive rows.
        min_rows (int): Minimum number of rows required for a trip to be considered valid.

    Returns:
        List of valid trip identifiers.
    """
    # Ensure required columns exist
    if trip_column not in data_df.columns or time_column not in data_df.columns:
        raise ValueError(f"Data sheet must contain '{trip_column}' and '{time_column}' columns.")

    # Sort data by trip and time to ensure chronological order
    data_sorted = data_df.sort_values(by=[trip_column, time_column]).reset_index(drop=True)

    # Group by trip and calculate time differences
    data_sorted['Time_Diff'] = data_sorted.groupby(trip_column)[time_column].diff()

    # Calculate the number of rows per trip
    trip_counts = data_sorted.groupby(trip_column).size().reset_index(name='Row_Count')

    # Identify trips where all time differences are within the specified range
    # and the number of rows is greater than min_rows
    trip_validity = data_sorted.groupby(trip_column).agg({
        'Time_Diff': lambda x: x.dropna().between(min_interval, max_interval).all()
    }).reset_index()

    # Merge with trip_counts to include Row_Count
    trip_validity = trip_validity.merge(trip_counts, on=trip_column)

    # Define validity based on both time differences and row count
    trip_validity['IsValid'] = trip_validity.apply(
        lambda row: row['Time_Diff'] and row['Row_Count'] > min_rows, axis=1
    )

    # Filter trips that are valid
    valid_trips = trip_validity[trip_validity['IsValid']][trip_column].tolist()

    return valid_trips


def filter_report(report_df, valid_trips, trip_column='report_trip_number'):
    """
    Filters the Report DataFrame to include only valid trips.

    Parameters:
        report_df (pd.DataFrame): DataFrame containing trip summaries.
        valid_trips (List): List of valid trip identifiers.
        trip_column (str): Column name in the Report sheet that contains trip identifiers.

    Returns:
        Filtered DataFrame with only valid trips.
    """
    if trip_column not in report_df.columns:
        raise ValueError(f"Report sheet must contain '{trip_column}' column.")

    # Split the report_trip_number by '_' and take the second part as trip number
    # Example: 'trip_1' -> 1
    def extract_trip_number(report_trip):
        try:
            parts = report_trip.split('_')
            if len(parts) < 2:
                raise ValueError(f"Invalid format: '{report_trip}'")
            trip_num = int(parts[1])
            return trip_num
        except (IndexError, ValueError) as e:
            print(f"Warning: Could not extract trip number from '{report_trip}'. Skipping this entry.")
            return pd.NA

    report_df['trip_numeric'] = report_df[trip_column].apply(extract_trip_number)

    # Identify and report rows with NaN trip_numeric
    nan_trips = report_df[report_df['trip_numeric'].isna()]
    if not nan_trips.empty:
        print(f"Warning: {len(nan_trips)} trip(s) in Report sheet have invalid format and will be skipped.")

    # Filter based on valid_trips and ensure trip_numeric is not NaN
    filtered_report = report_df[
        report_df['trip_numeric'].isin(valid_trips) & report_df['trip_numeric'].notna()
        ].reset_index(drop=True)

    return filtered_report


def create_boxplots(report_df, trip_column='report_trip_number', output_dir=None):
    """
    Creates boxplots for each numerical column in the Report DataFrame.

    Parameters:
        report_df (pd.DataFrame): Filtered Report DataFrame with valid trips.
        trip_column (str): Column name that contains trip identifiers.
        output_dir (str, optional): Directory to save the plots. If None, plots are shown.
    """
    # Identify numerical columns (excluding the trip identifier column)
    numerical_cols = report_df.select_dtypes(include=['number']).columns.tolist()

    # Remove the numeric trip identifier column if present
    if 'trip_numeric' in numerical_cols:
        numerical_cols.remove('trip_numeric')

    for col in numerical_cols:
        plt.figure(figsize=(8, 6))
        sns.boxplot(y=report_df[col])
        plt.title(f'Boxplot of {col} for Valid Trips')
        plt.ylabel(col)
        plt.tight_layout()

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            plot_path = os.path.join(output_dir, f"boxplot_{col}.png")
            plt.savefig(plot_path)
            plt.close()
            print(f"Boxplot for '{col}' saved to '{plot_path}'")
        else:
            plt.show()


def calculate_slope(trip_data, altitude_col='altitude', mileage_col='Cumulative_mileage', chunk_size=100):
    """
    Calculates the slope for each set of 100 consecutive rows in trip_data and assigns it to a new 'slope' column.

    Parameters:
        trip_data (pd.DataFrame): DataFrame containing data for a single trip.
        altitude_col (str): Name of the column representing altitude.
        mileage_col (str): Name of the column representing cumulative mileage.
        chunk_size (int): Number of rows in each chunk to calculate slope.

    Returns:
        pd.DataFrame: DataFrame with an additional 'slope' column.
    """
    # Initialize 'slope' column with NaN
    trip_data['slope'] = pd.NA

    total_rows = len(trip_data)

    # Iterate over the DataFrame in chunks of 100 rows
    for i in range(0, total_rows - chunk_size, chunk_size):
        start_altitude = trip_data.iloc[i][altitude_col]
        end_altitude = trip_data.iloc[i + chunk_size][altitude_col]
        start_mileage = trip_data.iloc[i][mileage_col]
        end_mileage = trip_data.iloc[i + chunk_size][mileage_col]

        mileage_diff = end_mileage - start_mileage

        if mileage_diff != 0:
            slope = (end_altitude - start_altitude) / mileage_diff / 10
            trip_data.loc[i:i + chunk_size - 1, 'slope'] = slope
        else:
            # If mileage_diff is zero, assign NaN
            trip_data.loc[i:i + chunk_size - 1, 'slope'] = pd.NA

    return trip_data


def save_valid_trips_as_csv(data_df, valid_trips, output_dir, trip_column='trip', altitude_col='altitude',
                            mileage_col='Cumulative_mileage'):
    """
    Saves each valid trip's data as a separate CSV file, adding a 'slope' column.

    Parameters:
        data_df (pd.DataFrame): Original Data DataFrame containing all trips.
        valid_trips (List): List of valid trip identifiers.
        output_dir (str): Directory to save the CSV files.
        trip_column (str): Column name that contains trip identifiers.
        altitude_col (str): Column name for altitude.
        mileage_col (str): Column name for cumulative mileage.
    """
    if trip_column not in data_df.columns:
        raise ValueError(f"Data sheet must contain '{trip_column}' column.")

    os.makedirs(output_dir, exist_ok=True)

    for trip in valid_trips:
        trip_data = data_df[data_df[trip_column] == trip].reset_index(drop=True)

        if trip_data.empty:
            print(f"Warning: No data found for valid trip '{trip}'. Skipping CSV export.")
            continue

        # Check if 'altitude' and 'Cumulative_mileage' columns exist
        if altitude_col not in trip_data.columns or mileage_col not in trip_data.columns:
            print(
                f"Warning: Trip '{trip}' missing '{altitude_col}' or '{mileage_col}' columns. Skipping slope calculation.")
            trip_data['slope'] = pd.NA
        else:
            # Calculate and assign 'slope' column
            trip_data = calculate_slope(trip_data, altitude_col, mileage_col, chunk_size=100)

        # Sanitize trip identifier for filename (remove or replace problematic characters)
        trip_safe = str(trip).replace('/', '_').replace('\\', '_').replace(':', '_').replace('*', '_') \
            .replace('?', '_').replace('"', '_').replace('<', '_').replace('>', '_') \
            .replace('|', '_')
        filename = f"Trip_{trip_safe}.csv"
        filepath = os.path.join(output_dir, filename)

        # Prevent overwriting existing files by appending a counter if necessary
        counter = 1
        base_filename, ext = os.path.splitext(filename)
        while os.path.exists(filepath):
            filepath = os.path.join(output_dir, f"{base_filename}_{counter}{ext}")
            counter += 1

        trip_data.to_csv(filepath, index=False)
        print(f"Valid trip '{trip}' saved to '{filepath}'")


def create_summary_report(report_df, trip_column='report_trip_number', output_dir=None):
    """
    Optional: Create a summary CSV of valid trips from the Report sheet.

    Parameters:
        report_df (pd.DataFrame): Filtered Report DataFrame with valid trips.
        trip_column (str): Column name that contains trip identifiers.
        output_dir (str, optional): Directory to save the summary CSV. If None, does not save.
    """
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        summary_path = os.path.join(output_dir, "filtered_report_summary.csv")
        report_df.to_csv(summary_path, index=False)
        print(f"Filtered Report summary saved to '{summary_path}'")


def main():
    # Path to your Excel file
    # excel_file = 'Qani_datas.xlsx'
    excel_file = r"C:\Users\s_alizadehnia\Downloads\Car_Dena_ef7tc_mt6_33 m 791 - ir 33_Data_10_22_2024, 01_00_00_to_10_29_2024, 08_20_00.xlsx"
    # Sheet names
    data_sheet = 'Data'
    report_sheet = 'Report'

    # Column names
    data_trip_column = 'trip'
    data_time_column = 'time'
    report_trip_column = 'report_trip_number'

    # Columns required for slope calculation
    altitude_col = 'altitude'
    mileage_col = 'Cumulative_mileage'

    # Output directories
    plots_output_dir = 'boxplots'  # Directory to save boxplots
    csv_output_dir = 'valid_trips_csv'  # Directory to save valid trip CSV files
    summary_output_dir = 'summary_reports'  # Directory to save summary reports (optional)

    plots_output_dir = 'C:/Users/s_alizadehnia/Desktop/LSTMPredictions/Data/Ehsan_955/Qani_data/test_qani_data1/'  # Directory to save boxplots
    csv_output_dir = 'C:/Users/s_alizadehnia/Desktop/LSTMPredictions/Data/Ehsan_955/Qani_data/test_qani_data1/'  # Directory to save valid trip CSV files
    summary_output_dir = 'C:/Users/s_alizadehnia/Desktop/LSTMPredictions/Data/Ehsan_955/Qani_data/test_qani_data1/'  # Directory to save summary reports (optional)
    # Read the Excel file
    try:
        data_df = pd.read_excel(excel_file, sheet_name=data_sheet)
        report_df = pd.read_excel(excel_file, sheet_name=report_sheet)
    except FileNotFoundError:
        print(f"Error: The file '{excel_file}' was not found.")
        return
    except ValueError as e:
        print(f"Error: {e}")
        return

    # Check if required columns exist in Data sheet
    if data_trip_column not in data_df.columns or data_time_column not in data_df.columns:
        print(f"Error: '{data_sheet}' sheet must contain '{data_trip_column}' and '{data_time_column}' columns.")
        return

    # Check if required column exists in Report sheet
    if report_trip_column not in report_df.columns:
        print(f"Error: '{report_sheet}' sheet must contain '{report_trip_column}' column.")
        return

    # Identify valid trips
    try:
        valid_trips = identify_valid_trips(
            data_df,
            trip_column=data_trip_column,
            time_column=data_time_column,
            min_interval=400,
            max_interval=2000,
            min_rows=600  # New condition: minimum number of rows
        )
    except ValueError as ve:
        print(f"Error during trip validation: {ve}")
        return

    print(f"Number of valid trips: {len(valid_trips)}")

    if not valid_trips:
        print("No valid trips found based on the given time interval and row count conditions.")
        return

    # Filter the Report sheet
    try:
        filtered_report = filter_report(
            report_df,
            valid_trips,
            trip_column=report_trip_column
        )
    except ValueError as ve:
        print(f"Error during Report sheet filtering: {ve}")
        return

    print(f"Filtered Report has {filtered_report.shape[0]} rows.")

    if filtered_report.empty:
        print("No matching trips found in the Report sheet for the valid trips identified.")
        return

    # Create boxplots
    create_boxplots(
        filtered_report,
        trip_column=report_trip_column,
        output_dir=plots_output_dir
    )

    # Save each valid trip as a separate CSV file, including the 'slope' column
    save_valid_trips_as_csv(
        data_df,
        valid_trips,
        output_dir=csv_output_dir,
        trip_column=data_trip_column,
        altitude_col=altitude_col,
        mileage_col=mileage_col
    )

    # Optional: Save a summary of the filtered Report sheet
    create_summary_report(
        filtered_report,
        trip_column=report_trip_column,
        output_dir=summary_output_dir
    )

    print("Processing complete.")


if __name__ == "__main__":
    main()
