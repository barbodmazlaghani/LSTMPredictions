import os
import pandas as pd
import glob

# Function to detect and fix jumps in the series (both positive and negative)
def fix_jumps(series, threshold=10000):
    """
    Detects and fixes positive and negative jumps in a pandas Series.

    Parameters:
        series (pd.Series): The data series to process.
        threshold (float): The minimum difference to consider as a positive jump.

    Returns:
        adjusted_series (pd.Series): The series after adjusting jumps.
        jumps_detected (bool): Whether any jumps were detected.
        jump_details (list): List of tuples containing jump information (index, type, difference).
    """
    adjusted_series = series.copy()
    jumps_detected = False
    jump_details = []

    i = 1
    while i < len(adjusted_series):
        current = adjusted_series[i]
        previous = adjusted_series[i - 1]

        # Negative Jump Detection
        if current < previous:
            diff = previous - current
            adjusted_series[i:] += diff
            jump_details.append((i, "Negative", diff))
            jumps_detected = True
            print(f"Detected Negative Jump at index {i}: {previous} -> {current} (diff={diff})")
            # No need to adjust 'current' as it's already handled by adding diff
        # Positive Jump Detection
        elif (current - previous) > threshold:
            diff = current - previous
            adjusted_series[i:] -= diff
            jump_details.append((i, "Positive", diff))
            jumps_detected = True
            print(f"Detected Positive Jump at index {i}: {previous} -> {current} (diff={diff})")

        # Move to the next element
        i += 1

    return adjusted_series, jumps_detected, jump_details

# Function to process a single CSV file and adjust Trip_fuel_consumption
def adjust_fuel_consumption(file_path, input_folder, output_folder, threshold=10000):
    """
    Processes a single CSV file to detect and fix jumps in 'Trip_fuel_consumption' column.
    Saves the processed file to the output folder.

    Parameters:
        file_path (str): Path to the input CSV file.
        input_folder (str): Root input folder path.
        output_folder (str): Root output folder path.
        threshold (float): Threshold for detecting positive jumps.
    """
    try:
        # Read the CSV file
        df = pd.read_csv(file_path)

        # Check if the required column is in the file
        if 'Trip_fuel_consumption' in df.columns:
            # Fix jumps and get the adjusted series
            adjusted_series, jumps_detected, jump_details = fix_jumps(df['Trip_fuel_consumption'], threshold)

            # Determine the relative path of the file to the input folder
            relative_path = os.path.relpath(file_path, input_folder)
            relative_dir = os.path.dirname(relative_path)
            filename = os.path.basename(relative_path)

            # Determine the output directory path
            output_dir = os.path.join(output_folder, relative_dir)

            # Ensure the output directory exists
            os.makedirs(output_dir, exist_ok=True)

            if jumps_detected:
                # Modify the filename to append '_fixed' before '.csv'
                name, ext = os.path.splitext(filename)
                new_filename = f"{name}{ext}"
                # Update the DataFrame with adjusted series
                df['Trip_fuel_consumption'] = adjusted_series
                # Determine the full output file path
                new_file_path = os.path.join(output_dir, new_filename)
                # Save the modified DataFrame to the new file
                df.to_csv(new_file_path, index=False)
                print(f"Jumps detected and fixed in file: {file_path}. Saved as: {new_file_path}")
                for idx, jump_type, diff in jump_details:
                    print(f"  - {jump_type} jump of {diff} at index {idx}")
            else:
                # No jumps detected, save the original DataFrame to output folder with original name
                new_file_path = os.path.join(output_dir, filename)
                df.to_csv(new_file_path, index=False)
                print(f"No jumps detected in file: {file_path}. Saved as: {new_file_path}")
        else:
            print(f"Column 'Trip_fuel_consumption' not found in {file_path}")
    except Exception as e:
        print(f"Error processing {file_path}: {e}")

# Function to process all CSV files in the folder and subfolders
def process_folder(input_folder, output_folder, threshold=10000):
    """
    Processes all CSV files in the input folder and its subfolders.
    Detects and fixes jumps in 'Trip_fuel_consumption' and saves the results to the output folder.

    Parameters:
        input_folder (str): Root input folder path.
        output_folder (str): Root output folder path.
        threshold (float): Threshold for detecting positive jumps.
    """
    # Get all CSV files recursively
    csv_files = glob.glob(os.path.join(input_folder, '**', '*.csv'), recursive=True)

    # Loop through each file and adjust fuel consumption
    for file_path in csv_files:
        adjust_fuel_consumption(file_path, input_folder, output_folder, threshold)

# Example usage
if __name__ == "__main__":
    # Define your input and output folders
    input_folder = 'C:/Users/s_alizadehnia/Desktop/LSTMPredictions/Data/Ehsan_955/source/generated'  # Replace with your input folder path
    output_folder = 'C:/Users/s_alizadehnia/Desktop/LSTMPredictions/Data/Ehsan_955/source/processed'  # Replace with your desired output folder path

    # Optional: Define the threshold for positive jumps
    jump_threshold = 10000  # Adjust this value based on your data

    # Process all files
    process_folder(input_folder, output_folder, threshold=jump_threshold)
