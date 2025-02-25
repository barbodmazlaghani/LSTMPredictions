import os
import streamlit as st
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
from torch import nn

# Constants
SEQUENCE_LENGTH = 60
PLOT_SAVE_DIR = 'predicted_vs_actual_plots'
# DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Ensure the base save directory exists
os.makedirs(PLOT_SAVE_DIR, exist_ok=True)

# Define the PyTorch model structure
class FuelConsumptionModel(nn.Module):
    def __init__(self, input_size):
        super(FuelConsumptionModel, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.fc(x)

# Load the trained model
def load_trained_model(model_path, input_size):
    model = FuelConsumptionModel(input_size=input_size)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

# Process the file and prepare segments
def process_file(file_path):
    df = pd.read_csv(file_path)
    if not df.empty:
        df.rename(columns={
            'Vehicle_Speed': 'Speed',
            'Trip_fuel_consumption': 'Trip fuel consumption',
            'Coolant_temp': 'Coolant_temperature'
        }, inplace=True)

        df['Momentary fuel consumption1'] = df['Trip fuel consumption'].diff().fillna(0)
        df['Momentary fuel consumption2'] = df['Trip fuel consumption'].diff().shift(-1).fillna(0)
        df['Acceleration1'] = df['Speed'].diff().fillna(0)
        df['Acceleration2'] = df['Speed'].diff().shift(-1).fillna(0)
        df['Current_gear_shift_position_(Current_gear)'] = df['Current_gear_shift_position_(Current_gear)'].replace({13: 0, 14: 1})

        features = df[['Acceleration1', 'Acceleration2', 'Speed', 'Current_gear_shift_position_(Current_gear)', 'slope']]
        target = df['Momentary fuel consumption2']

    return features, target, df

# Pad and normalize the data
def pad_and_normalize(data, sequence_length=SEQUENCE_LENGTH):
    padded_data = np.zeros((len(data), sequence_length, data[0].shape[1]))
    for i, seq in enumerate(data):
        length = min(len(seq), sequence_length)
        padded_data[i, :length] = seq[:length]

    min_val_x = [-10, -10, 0, 0, -10]
    max_val_x = [10, 10, 200, 5, 10]

    for i in range(padded_data.shape[-1]):
        padded_data[:, :, i] = (padded_data[:, :, i] - min_val_x[i]) / (max_val_x[i] - min_val_x[i])

    return torch.tensor(padded_data, dtype=torch.float32).to(DEVICE)

# Predict and plot the results
def plot_predicted_vs_real(input_file, model, model_name):
    features, actual_values, df1 = process_file(input_file)

    mileagee = (df1['Cumulative_mileage'].iloc[-1] - df1['Cumulative_mileage'].iloc[0])
    trip_fuel_consp100 = (((df1['Trip fuel consumption'].iloc[-1] - df1['Trip fuel consumption'].iloc[0]) / 10000) / mileagee)

    num_segments = len(features) // SEQUENCE_LENGTH
    predictions = []

    for i in range(num_segments):
        segment = features.iloc[i * SEQUENCE_LENGTH:(i + 1) * SEQUENCE_LENGTH]
        segment_normalized = pad_and_normalize([segment.values])
        with torch.no_grad():
            segment_predictions = model(segment_normalized).cpu().numpy()
        predictions.extend(segment_predictions.flatten() * 30000)

    remainder = len(features) % SEQUENCE_LENGTH
    if remainder != 0:
        last_segment = features.iloc[-remainder:]
        last_segment_normalized = pad_and_normalize([last_segment.values], sequence_length=remainder)
        with torch.no_grad():
            last_segment_predictions = model(last_segment_normalized).cpu().numpy()
        predictions.extend(last_segment_predictions.flatten() * 30000)

    predictions = np.array(predictions)
    actual_valuess = actual_values.values[:len(predictions)]

    mae = mean_absolute_error(actual_valuess, predictions)
    mse = mean_squared_error(actual_valuess, predictions)

    non_zero_actual = actual_valuess != 0
    mape = np.mean(np.abs((actual_valuess[non_zero_actual] - predictions[non_zero_actual]) / actual_valuess[non_zero_actual])) * 100

    trip_fuel_consp100_p = np.cumsum(predictions[:len(actual_values)], axis=0)
    trip_fuel_cons_predicted_p100 = (trip_fuel_consp100_p[-1] / 10000) / mileagee

    plt.figure(figsize=(10, 6))
    plt.plot(np.cumsum(actual_values.values[:len(predictions)], axis=0), label='Real', color='blue')
    plt.plot(np.cumsum(predictions[:len(actual_values)], axis=0), label='Predicted', color='red')
    plt.xlabel('Index')
    plt.ylabel('Fuel Consumption')
    plt.title(f'Predicted vs Real Fuel Consumption ({model_name})')
    plt.legend()

    text = (
        f"MAE: {mae:.4f}\n"
        f"MSE: {mse:.4f}\n"
        f"MAPE: {mape:.2f}%\n"
    )
    plt.text(0.95, 0.05, text, fontsize=10, color='gray', horizontalalignment='right', verticalalignment='bottom', transform=plt.gca().transAxes)

    st.pyplot(plt)

# Streamlit UI
st.title('Fuel Consumption Prediction')

uploaded_csv = st.file_uploader('Upload CSV File', type='csv')
uploaded_model = st.file_uploader('Upload Model File', type='pth')

if uploaded_csv and uploaded_model:
    csv_path = f"temp_{uploaded_csv.name}"
    model_path = f"temp_{uploaded_model.name}"

    with open(csv_path, 'wb') as f:
        f.write(uploaded_csv.getbuffer())

    with open(model_path, 'wb') as f:
        f.write(uploaded_model.getbuffer())

    model_name = os.path.splitext(uploaded_model.name)[0]
    model = load_trained_model(model_path, input_size=5)

    st.write('Processing and Plotting Results...')
    plot_predicted_vs_real(csv_path, model, model_name)