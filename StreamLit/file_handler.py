import os
import pandas as pd
import numpy as np

def handle_uploaded_file(uploaded_file):
    """
    Reads the uploaded file and returns a DataFrame.
    """
    if uploaded_file.name.endswith('.csv'):
        return pd.read_csv(uploaded_file)
    else:
        return pd.read_excel(uploaded_file)

def save_to_npz(folder, filename, data):
    """
    Saves data to NPZ format.
    """
    npz_path = os.path.join(folder, f"{os.path.splitext(filename)[0]}.npz")
    np.savez_compressed(npz_path, data=data)
    return npz_path
