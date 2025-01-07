import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
from file_handler import save_to_npz

def preprocess_data(df, file_name, save_folder="processed_files"):
    """
    Normalizes the data and saves it to NPZ format.
    
    Args:
    - df: DataFrame to process.
    - file_name: Original file name of the uploaded data.
    - save_folder: Directory where processed data is saved.
    
    Returns:
    - normalized_data: Normalized numpy array of the data.
    - npz_path: Path to the saved NPZ file.
    """
    os.makedirs(save_folder, exist_ok=True)
    data = df.values

    # Normalize data
    scaler = MinMaxScaler()
    normalized_data = scaler.fit_transform(data)

    # Save to NPZ
    npz_path = save_to_npz(save_folder, file_name, normalized_data)
    return normalized_data, npz_path
