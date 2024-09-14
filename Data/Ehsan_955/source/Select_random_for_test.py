import os
import random
import shutil


# Function to randomly select, copy, and delete 20 CSV files
def random_select_copy_and_delete_csv(source_folder, target_folder, num_files=20):
    # Ensure the target folder exists
    os.makedirs(target_folder, exist_ok=True)

    # Get all CSV files from the source folder
    csv_files = [f for f in os.listdir(source_folder) if f.endswith('.csv')]

    # Check if there are enough files to choose from
    if len(csv_files) < num_files:
        print(f"Warning: Only {len(csv_files)} CSV files available, fewer than {num_files} requested.")
        num_files = len(csv_files)  # Adjust to available file count

    # Randomly select the number of files requested
    selected_files = random.sample(csv_files, num_files)

    # Copy the selected files to the target folder and delete them from the source folder
    for file_name in selected_files:
        source_file_path = os.path.join(source_folder, file_name)
        target_file_path = os.path.join(target_folder, file_name)

        # Copy the file to the target folder
        shutil.copy(source_file_path, target_file_path)
        print(f"Copied: {file_name} to {target_folder}")

        # Delete the file from the source folder
        os.remove(source_file_path)
        print(f"Deleted: {file_name} from {source_folder}")

    print(f"Selected {num_files} CSV files, copied them to {target_folder}, and deleted them from {source_folder}")


# Example usage
source_folder = 'C:/Users/s_alizadehnia/Desktop/LSTMPredictions/Data/Ehsan_955/source/processed'  # Replace with your source folder path
target_folder = 'C:/Users/s_alizadehnia/Desktop/LSTMPredictions/Data/Ehsan_955/source/test'  # Replace with your target folder path

# Call the function to randomly select, copy, and delete 20 files
random_select_copy_and_delete_csv(source_folder, target_folder, num_files=20)

