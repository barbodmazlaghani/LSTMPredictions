import os
import pandas as pd
import streamlit as st

SEQUENCE_LENGTH = 40

# Data processing function (optimized)
def process_file(file_path, selected_columns):
    df = pd.read_csv(file_path)
    if df.empty:
        return None, None

    df[selected_columns["time_column"]] -= df[selected_columns["time_column"]].iloc[0]  # Adjust the time to start from 0
    df['Momentary fuel consumption1'] = df[selected_columns["fuel_column"]].diff().fillna(0)
    df['Momentary fuel consumption2'] = df[selected_columns["fuel_column"]].diff().shift(-1).fillna(0)
    df['Acceleration1'] = df[selected_columns["speed_column"]].diff().fillna(0)
    df['Acceleration2'] = df[selected_columns["speed_column"]].diff().shift(-1).fillna(0)
    df[selected_columns["gear_column"]] = df[selected_columns["gear_column"]].replace({13: 0, 14: 1})

    df = df.iloc[:-5]

    features = df[['Acceleration1', 'Acceleration2', selected_columns["speed_column"], selected_columns["gear_column"], 'slope']].values
    target = df['Momentary fuel consumption2'].values

    return features, target

# Function to slice data into chunks
def slice_data(features, target, sequence_length):
    num_chunks = len(features) // sequence_length
    X_list, y_list = [], []

    for i in range(num_chunks):
        start_idx = i * sequence_length
        end_idx = start_idx + sequence_length
        sliced_features = features[start_idx:end_idx]
        sliced_target = target[start_idx:end_idx]
        X_list.append(sliced_features)
        y_list.append(sliced_target)

    return X_list, y_list

# Function to process file and slice it
def process_file_and_slice(file_path, selected_columns):
    features, target = process_file(file_path, selected_columns)
    if features is not None and target is not None:
        return slice_data(features, target, SEQUENCE_LENGTH)
    return [], []

# Sequential processing of the folder with progress bar
def process_folder_with_progress(folder_path, sequence_length, selected_columns):
    all_X, all_y = [], []
    file_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.csv')]

    # Initialize Streamlit progress bar
    progress_bar = st.progress(0)
    progress_text = st.empty()

    total_files = len(file_paths)
    for idx, file_path in enumerate(file_paths):
        X_list, y_list = process_file_and_slice(file_path, selected_columns)
        all_X.extend(X_list)
        all_y.extend(y_list)

        # Update progress
        progress_percentage = int(((idx + 1) / total_files) * 100)
        progress_bar.progress(progress_percentage)
        progress_text.text(f"Processing files: {idx + 1}/{total_files}")

    progress_text.text("Processing completed!")
    return all_X, all_y

# Main function to call the processing logic
def augment(folder_path, selected_columns):
    st.title("Sequential File Processing with Progress Bar")

    with st.spinner("Processing..."):
        X_augmented, y_augmented = process_folder_with_progress(folder_path, SEQUENCE_LENGTH, selected_columns)
        st.success("Processing completed!")
        st.write(f"Total sequences processed: {len(X_augmented)}")
        return X_augmented, y_augmented, SEQUENCE_LENGTH
