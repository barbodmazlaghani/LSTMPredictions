import os
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import LSTM, Dense, Bidirectional
# from tensorflow.keras.models import load_model

# Helper functions
def save_to_npz(folder, filename, data):
    npz_path = os.path.join(folder, filename.replace('.csv', '.npz'))
    np.savez(npz_path, data=data)
    return npz_path

def normalize_data(data):
    scaler = MinMaxScaler()
    return scaler, scaler.fit_transform(data)

def create_bilstm_model(input_shape):
    pass
    # return model

def train_model(model, X_train, y_train, epochs=10, batch_size=32):
    # model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)
    pass
    # return model

# Streamlit App
st.title("BiLSTM Model Trainer and Tester")

# Upload initial file
st.header("1. Upload CSV/XLSX for Processing")
uploaded_file = st.file_uploader("Upload your file", type=['csv', 'xlsx'])


if uploaded_file:
    # Read the file
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)
    st.write("Uploaded Data:", df.head())

    # Folder creation
    folder = "processed_files"
    os.makedirs(folder, exist_ok=True)
    csv_path = os.path.join(folder, uploaded_file.name)
    df.to_csv(csv_path, index=False)
    st.success(f"File saved to {folder}.")

    # Convert to NPZ

    data = df.values
    npz_path = save_to_npz(folder, uploaded_file.name, data)
    st.success(f"Converted to NPZ format: {npz_path}")

    # Normalize and Train Model
    st.header("2. Normalize and Train Model")
    scaler, normalized_data = normalize_data(data)
    X_train = normalized_data[:, :-1]  # Features
    y_train = normalized_data[:, -1]   # Labels

    model = create_bilstm_model((X_train.shape[1], 1))
    model = train_model(model, X_train, y_train)

    model_path = os.path.join(folder, "bilstm_model.h5")
    model.save(model_path)
    st.success(f"Model trained and saved to {model_path}.")

# Upload for testing
st.header("3. Test the Model")
model_file = st.file_uploader("Upload Model Weights (.h5)", type=['h5'])
test_file = st.file_uploader("Upload Test CSV", type=['csv'])

if model_file and test_file:
    # Load the model
    # model = load_model(model_file)
    test_df = pd.read_csv(test_file)
    st.write("Test Data:", test_df.head())

    # Normalize test data
    _, normalized_test_data = normalize_data(test_df.values)
    X_test = normalized_test_data[:, :-1]
    y_test = normalized_test_data[:, -1]

    # Predictions
    # predictions = model.predict(X_test)
    st.write("Predictions:", predictions)

    # Visualization
    st.line_chart(predictions)
    st.write("Actual vs Predicted:")
    result_df = pd.DataFrame({"Actual": y_test.flatten(), "Predicted": predictions.flatten()})
    st.write(result_df)
