import os
import streamlit as st
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from altitude_processor import process_altitudes
from slope_calculator import process_files_with_slope
from file_handler import  handle_uploaded_file  # Assuming these are implemented
import shutil  

st.set_option('client.showErrorDetails', False)

# Constants
SEQUENCE_LENGTH = 600
PLOT_SAVE_DIR = 'predicted_vs_actual_plots'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Ensure the base save directory exists
os.makedirs(PLOT_SAVE_DIR, exist_ok=True)


# Define the PyTorch model structure
class FuelConsumptionModel(nn.Module):
    def __init__(self, input_size):
        super(FuelConsumptionModel, self).__init__()
        # LSTM layers
        self.lstm1 = nn.LSTM(input_size, 32, batch_first=True, bidirectional=True)
        self.lstm2 = nn.LSTM(64, 32, batch_first=True, bidirectional=True)

        # Layer Normalization after each LSTM
        self.layer_norm1 = nn.LayerNorm(64)
        self.layer_norm2 = nn.LayerNorm(64)

        # Dropout layers
        self.dropout1 = nn.Dropout(0.2)
        self.dropout2 = nn.Dropout(0.2)

        # Dense output layer
        self.dense = nn.Linear(64, 1)

    def forward(self, x):
        # LSTM 1 + Layer Normalization + Dropout
        x, _ = self.lstm1(x)
        x = self.layer_norm1(x)  # Apply layer normalization after LSTM1
        x = self.dropout1(x)

        # LSTM 2 + Layer Normalization + Dropout
        x, _ = self.lstm2(x)
        x = self.layer_norm2(x)  # Apply layer normalization after LSTM2
        x = self.dropout2(x)

        # Dense layer for final output
        x = self.dense(x)

        return x

# Load the trained model
def load_trained_model(model_path, input_size):
    model = FuelConsumptionModel(input_size=input_size)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

# Process the file and prepare segments
def process_file(df ):
    
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
        df['Current_gear_shift_position_(Current_gear)'] = df['Current_gear_shift_position_(Current_gear)'].replace({13: 0, 14: 1,'n': 0,'N': 0})

        features = df[['Acceleration1', 'Acceleration2', 'Speed', 'Current_gear_shift_position_(Current_gear)', 'slope']]
        target = df['Momentary fuel consumption2']

    return features, target, df

# Pad and normalize the data
def pad_and_normalize(data, sequence_length=SEQUENCE_LENGTH):
    padded_data = np.zeros((len(data), sequence_length, data[0].shape[1]))
    for i, seq in enumerate(data):
        length = min(len(seq), sequence_length)
        padded_data[i, :length] = seq[:length]

    min_val_x = [-10, -10,0, 0, -10]
    max_val_x = [10, 10, 200, 5, 10]

    for i in range(padded_data.shape[-1]):
        padded_data[:, :, i] = (padded_data[:, :, i] - min_val_x[i]) / (max_val_x[i] - min_val_x[i])

    return torch.tensor(padded_data, dtype=torch.float32).to(DEVICE)






def plot_predicted_vs_real(df, model, model_name ,selected_columns):
    features, actual_values, df1 = process_file(df)

    timee = (df1['time'].iloc[-1] - df1['time'].iloc[0]) / 60000
    mileagee = (df1['Cumulative_mileage'].iloc[-1] - df1['Cumulative_mileage'].iloc[0])

    if df1['Trip fuel consumption'].iloc[-1]/100000 < 1:
        trip_fuel_consp100 = (((df1['Trip fuel consumption'].iloc[-1] - df1['Trip fuel consumption'].iloc[0]) *100) / mileagee)
        num_segments = len(features) // SEQUENCE_LENGTH
        predictions = []

        for i in range(num_segments):
            segment = features.iloc[i * SEQUENCE_LENGTH:(i + 1) * SEQUENCE_LENGTH]
            segment_normalized = pad_and_normalize([segment.values])
            with torch.no_grad():
                segment_predictions = model(segment_normalized).cpu().numpy()
            predictions.extend(segment_predictions.flatten() *30000)

        remainder = len(features) % SEQUENCE_LENGTH
        if remainder != 0:
            last_segment = features.iloc[-remainder:]
            last_segment_normalized = pad_and_normalize([last_segment.values], sequence_length=remainder)
            with torch.no_grad():
                last_segment_predictions = model(last_segment_normalized).cpu().numpy()
            predictions.extend(last_segment_predictions.flatten())

        predictions = np.array(predictions)
        actual_valuess = actual_values.values[:len(predictions)]

        mae = mean_absolute_error(actual_valuess, predictions)
        mse = mean_squared_error(actual_valuess, predictions)

        non_zero_actual = actual_valuess != 0
        mape = np.mean(np.abs((actual_valuess[non_zero_actual] - predictions[non_zero_actual]) / actual_valuess[non_zero_actual])) * 100

        trip_fuel_consp100_p = np.cumsum(predictions[:len(actual_values)], axis=0)
        trip_fuel_cons_predicted_p100 = (trip_fuel_consp100_p[-1] *100) / mileagee

        # Create Plotly figure with subplots (cumulative and raw in separate plots)
        fig = make_subplots(
            rows=2, cols=1,  # Two rows, one column
            shared_xaxes=True,  # Keep the x-axes shared between both plots
            vertical_spacing=0.1,  # Add space between the plots
            subplot_titles=("Cumulative Fuel Consumption", "Raw Fuel Consumption"),
            row_heights=[0.6, 0.4]  # Adjust row heights
        )

        # Add real values trace (cumulative sum)
        fig.add_trace(go.Scatter(
            x=np.arange(len(actual_valuess)),
            y=np.cumsum(actual_valuess),
            mode='lines',
            name='Real (Cumulative)',
            line=dict(color='blue', dash='solid')
        ), row=1, col=1)

        # Add predicted values trace (cumulative sum)
        fig.add_trace(go.Scatter(
            x=np.arange(len(predictions)),
            y=np.cumsum(predictions),
            mode='lines',
            name='Predicted (Cumulative)',
            line=dict(color='red', dash='solid')
        ), row=1, col=1)

        # Add real values trace (raw values)
        fig.add_trace(go.Scatter(
            x=np.arange(len(actual_valuess)),
            y=actual_valuess,
            mode='lines',
            name='Real (Raw)',
            line=dict(color='blue', dash='dot')
        ), row=2, col=1)

        # Add predicted values trace (raw values)
        fig.add_trace(go.Scatter(
            x=np.arange(len(predictions)),
            y=predictions,
            mode='lines',
            name='Predicted (Raw)',
            line=dict(color='red', dash='dot')
        ), row=2, col=1)

        # Update layout for subplots
        fig.update_layout(
            
            xaxis_title="Index",
            yaxis_title="Fuel Consumption (Cumulative)",
            showlegend=True,
            height=600,  # Increase the height of the plot
            width=10000,  # Increase the width of the plot
            annotations=[
                dict(
                    x=0.5, y=0.95, xref='paper', yref='paper',
                    text=(
                        f"time(M): {timee:.0f}<br>"
                        f"mileage(KM): {mileagee:.1f}<br>"
                        f"Real fuel cons: {trip_fuel_consp100:.2f}<br>"
                        f"Pred fuel cons: {trip_fuel_cons_predicted_p100:.2f}<br>"
                        f"Error(%): {((abs(trip_fuel_consp100 - trip_fuel_cons_predicted_p100)) / trip_fuel_consp100) * 100:.2f}"
                    ),
                    showarrow=False,
                    font=dict(size=12, color="gray")
                )
            ]
        )

        # Show plot in Streamlit
        st.plotly_chart(fig, use_container_width=True)
    else:
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

        # Create Plotly figure with subplots (cumulative and raw in separate plots)
        fig = make_subplots(
            rows=2, cols=1,  # Two rows, one column
            shared_xaxes=True,  # Keep the x-axes shared between both plots
            vertical_spacing=0.1,  # Add space between the plots
            subplot_titles=("Cumulative Fuel Consumption", "Raw Fuel Consumption"),
            row_heights=[0.6, 0.4]  # Adjust row heights
        )

        # Add real values trace (cumulative sum)
        fig.add_trace(go.Scatter(
            x=np.arange(len(actual_valuess)),
            y=np.cumsum(actual_valuess),
            mode='lines',
            name='Real (Cumulative)',
            line=dict(color='blue', dash='solid')
        ), row=1, col=1)

        # Add predicted values trace (cumulative sum)
        fig.add_trace(go.Scatter(
            x=np.arange(len(predictions)),
            y=np.cumsum(predictions),
            mode='lines',
            name='Predicted (Cumulative)',
            line=dict(color='red', dash='solid')
        ), row=1, col=1)

        # Add real values trace (raw values)
        fig.add_trace(go.Scatter(
            x=np.arange(len(actual_valuess)),
            y=actual_valuess,
            mode='lines',
            name='Real (Raw)',
            line=dict(color='blue', dash='dot')
        ), row=2, col=1)

        # Add predicted values trace (raw values)
        fig.add_trace(go.Scatter(
            x=np.arange(len(predictions)),
            y=predictions,
            mode='lines',
            name='Predicted (Raw)',
            line=dict(color='red', dash='dot')
        ), row=2, col=1)

        # Update layout for subplots
        fig.update_layout(
            
            xaxis_title="Index",
            yaxis_title="Fuel Consumption (Cumulative)",
            showlegend=True,
            height=600,  # Increase the height of the plot
            width=10000,  # Increase the width of the plot
            annotations=[
                dict(
                    x=0.5, y=0.95, xref='paper', yref='paper',
                    text=(
                        f"time(M): {timee:.0f}<br>"
                        f"mileage(KM): {mileagee:.1f}<br>"
                        f"Real fuel cons: {trip_fuel_consp100:.2f}<br>"
                        f"Pred fuel cons: {trip_fuel_cons_predicted_p100:.2f}<br>"
                        f"Error(%): {((abs(trip_fuel_consp100 - trip_fuel_cons_predicted_p100)) / trip_fuel_consp100) * 100:.2f}"
                    ),
                    showarrow=False,
                    font=dict(size=12, color="gray")
                )
            ]
        )

        # Show plot in Streamlit
        st.plotly_chart(fig, use_container_width=True)






def is_monotonic_increasing_with_stability(time_series, fuel_series):
    """
    Ensure that time is strictly increasing, and fuel consumption either stays the same or increases.
    """
    return time_series.is_monotonic_increasing and all(fuel_series.diff().ge(0) | fuel_series.diff().eq(0))


@st.cache_data
def read_file(uploaded_file):
    """Handle file upload and caching."""
    return handle_uploaded_file(uploaded_file)


def show_fuel_plot():
    try:
        shutil.rmtree(temp_folder)  # حذف کل پوشه temp
    except Exception as e:
        print(f"Error while cleaning up temporary folder: {e}")
    st.title('Fuel Consumption Prediction')
    MODEL_DIR = "C:/Users/s_alizadehnia/Desktop/LSTMPredictions/StreamLit/models"
    uploaded_csv = st.file_uploader("Upload your file", type=['csv', 'xlsx'])
    # آپلود فایل‌ها
    # uploaded_csv = st.file_uploader('Upload CSV File', type='csv')
    existing_models = [None] + [f for f in os.listdir(MODEL_DIR) if f.endswith('.pth')]
    selected_model = None
    if existing_models:
        selected_model = st.selectbox(
            "Select a Pre-trained Model (If you didn't upload one):",
            options=existing_models
        )
    else:
        st.warning("No pre-trained models available. Please upload a model.")

    uploaded_model = st.file_uploader('Upload Model File', type='pth')
    
    if uploaded_csv and (uploaded_model or selected_model):
        temp_folder = "temp"
        os.makedirs(temp_folder, exist_ok=True)
        csv_path = os.path.join(temp_folder, f"temp_{uploaded_csv.name}")
        if uploaded_model:
            # اگر فایل مدل آپلود شده باشد، از آن استفاده می‌شود
            model_path = os.path.join(temp_folder, f"temp_{uploaded_model.name}")
            with open(model_path, 'wb') as f:
                f.write(uploaded_model.getbuffer())
            st.success(f"Uploaded model is being used: {uploaded_model.name}")
        else:
            # اگر فایلی آپلود نشده باشد، مدل انتخاب شده استفاده می‌شود
            model_path = os.path.join(MODEL_DIR, selected_model)
            st.success(f"Pre-trained model is being used: {selected_model}")
        
        # خواندن داده‌ها.
        df = read_file(uploaded_csv)
        # df = pd.read_csv(uploaded_csv)
        column_options = [None] + df.columns.tolist()


        # انتخاب ستون‌ها توسط کاربر
        st.subheader("Step 1: Select Required Columns")
        time_column = st.selectbox("Select the time Column", column_options, key="time_column")
        speed_column = st.selectbox("Select the Speed Column", column_options, key="speed_column")
        fuel_column = st.selectbox("Select the Trip Fuel Consumption Column", column_options, key="fuel_column")
        mileage_column = st.selectbox("Select the Cumulative Mileage Column", column_options, key="mileage_column")
        gear_column = st.selectbox("Select the Current Gear Column", column_options, key="gear_column")

        # اضافه کردن ستون‌های latitude و longitude
        lat_column = st.selectbox("Select the Latitude Column", column_options, key="lat_column")
        long_column = st.selectbox("Select the Longitude Column", column_options, key="long_column")
        selected_columns = {
                        'time_column': time_column,
                        'speed_column': speed_column,
                        'fuel_column': fuel_column,
                        'mileage_column': mileage_column,
                        'gear_column': gear_column,
                        'latitude_column': lat_column,
                        'longitude_column': long_column,
                    }
        # بررسی اینکه آیا سفر جدا شده است
        # 1. پرسش اینکه داده‌ها مربوط به یک سفر هستند یا خیر
        is_single_trip = st.radio(
            "Does the data belong to a single trip?",
            options=["Yes", "No"],
            index=0,  # پیش‌فرض "Yes"
        )

        # 2. در صورت چند سفری بودن، بررسی وجود ستون trip
        
        if is_single_trip == "No":
            trip_column = None
            engine_speed_column = None
            voltage_column = None
            has_trip_column = st.radio(
                "Do you have a column to identify trips?",
                options=["Yes", "No"],
                index=0,  # پیش‌فرض "Yes"
            )
            if has_trip_column == "Yes":
                trip_column = st.selectbox("Select the Trip Column", column_options, key="trip_column")
            else:
                engine_speed_column = st.selectbox("Select the Engine Speed Column", column_options, key="engine_speed")
                voltage_column = st.selectbox("Select the Battery Voltage Column", column_options, key="voltage")
                trip_column = None
            df = df[1:]  # حذف ردیف اول اگر حاوی هدر باشد
            
            # گرفتن ستون‌های جدید
            selected_columns = {
                            'time_column': time_column,
                            'speed_column': speed_column,
                            'fuel_column': fuel_column,
                            'mileage_column': mileage_column,
                            'gear_column': gear_column,
                            'latitude_column': lat_column,
                            'longitude_column': long_column,
                            'trip_column': trip_column,
                        }
            
            # اگر ستون trip وجود داشته باشد
            if trip_column:
                # تقسیم داده‌ها بر اساس ستون trip
                unique_trips = df[trip_column].unique()
                for trip_id in unique_trips:
                    trip_df = df[df[trip_column] == trip_id]
                    filename = f"{temp_folder}/trip_{trip_id}.csv"
                    trip_df.to_csv(filename, index=False)
                    print(f"Saved {filename}")

                try:
                    os.remove(trip_csv_path)
                except Exception as e:
                    print(f"Error while cleaning up temporary trip files: {e}")
            

            elif engine_speed_column and voltage_column:
                # اگر ستون trip وجود نداشته باشد، بر اساس Engine_speed و Voltage تقسیم می‌کنیم
                
                

                df[engine_speed_column] = pd.to_numeric(df[engine_speed_column], errors='coerce')
                df[voltage_column] = pd.to_numeric(df[voltage_column], errors='coerce')
                df[selected_columns["time_column"]] = pd.to_numeric(df[selected_columns["time_column"]], errors='coerce')
                df[selected_columns["fuel_column"]] = pd.to_numeric(df[selected_columns["fuel_column"]], errors='coerce')

                trip_number = 1
                trip_data = []
                current_trip = []

                for index, row in df.iterrows():
                    # print('1')
                    engine_speed = row[engine_speed_column]
                    battery_voltage = row[voltage_column]

                    if engine_speed > 0 and battery_voltage > 0:  # Start or continue a trip
                        # print('2')
                        current_trip.append(row)
                    else:  # End the current trip
                        # print('3')
                        if current_trip:
                            # print('4')
                            trip_data.append(current_trip)
                            current_trip = []

                # Save the last trip if exists
                if current_trip:
                    # print('5')
                    trip_data.append(current_trip)

                # Process each identified trip
                for idx, trip in enumerate(trip_data):
                    # print('6')
                    trip_df = pd.DataFrame(trip)

                    # Check monotonicity
                    if not is_monotonic_increasing_with_stability(trip_df[selected_columns["time_column"]], trip_df[selected_columns["fuel_column"]]):
                        # Split based on monotonicity
                        # print('7')
                        split_groups = []
                        current_group = []
                        last_time = None
                        last_fuel = None

                        for index, row in trip_df.iterrows():
                            # print('8')
                            if last_time is None or (row[selected_columns["time_column"]] >= last_time and (row[selected_columns["fuel_column"]] >= last_fuel or row[selected_columns["fuel_column"]] == last_fuel)):
                                current_group.append(row)
                                # print('9')
                            else:
                                # print('10')
                                split_groups.append(pd.DataFrame(current_group))
                                current_group = [row]
                            last_time = row[selected_columns["time_column"]]
                            last_fuel = row[selected_columns["fuel_column"]]

                        split_groups.append(pd.DataFrame(current_group))

                        # Save each split group
                        for i, split_group in enumerate(split_groups):
                            # print('11')
                            filename = f"{temp_folder}/part_{idx+1}_{i+1}.csv"
                            split_group.to_csv(filename, index=False)
                            print(f"Saved {filename}")
                    else:
                        print('12')
                        # Save monotonic trip
                        filename = f"{temp_folder}/trip_{trip_number}.csv"
                        trip_df.to_csv(filename, index=False)
                        print(f"Saved {filename}")
                        trip_number += 1
                
                

        # بررسی دکمه پردازش
        if st.button("Process and Predict"):
            if not uploaded_model and not selected_model:
                st.error("Please upload a model file.")
                return

            # ذخیره موقت فایل‌ها
            

            

            with open(model_path, 'wb') as f:
                f.write(uploaded_model.getbuffer())

            # استفاده از ستون‌های انتخاب‌شده
            
            process_altitudes(temp_folder, selected_columns)
            process_files_with_slope(temp_folder, selected_columns)

                    # پیش‌بینی مدل
            model_name = os.path.splitext(uploaded_model.name)[0]
            model = load_trained_model(model_path, input_size=5)

            # اگر داده‌ها برای یک سفر هستند
            if is_single_trip == "Yes":
                with open(csv_path, 'wb') as f:
                    f.write(uploaded_csv.getbuffer())
                st.write("Processing as a single trip...")
                # بازنویسی داده‌ها با ستون‌های انتخاب‌شده
                df.rename(columns={
                    time_column: 'time',
                    speed_column: 'Speed',
                    fuel_column: 'Trip fuel consumption',
                    mileage_column: 'Cumulative_mileage',
                    gear_column: 'Current_gear_shift_position_(Current_gear)',
                }, inplace=True)

                # Process altitudes with latitude and longitude
                # process_altitudes(temp_folder, selected_columns)

                # # Process files with slope
                # process_files_with_slope(temp_folder, selected_columns)

                # # پردازش مدل و پیش‌بینی
                # model_name = os.path.splitext(uploaded_model.name)[0]
                # model = load_trained_model(model_path, input_size=5)

                st.write('Processing and Plotting Results...')
                plot_predicted_vs_real(df, model, model_name ,selected_columns)

            # اگر داده‌ها برای چندین سفر هستند
            else:
                try:
                    os.remove(trip_csv_path)
                except Exception as e:
                    print(f"Error while cleaning up temporary trip files: {e}")
                # اگر داده‌ها برای چندین سفر هستند
                st.write("Processing for multiple trips...")
                
                # لیست تمام فایل‌های CSV در فولدر temp
                temp_files = [f for f in os.listdir(temp_folder) if f.endswith('.csv')]

                for file in temp_files:
                    trip_csv_path = os.path.join(temp_folder, file)
                    st.write(f"\n\nProcessing file: {file}\n\n")

                    # خواندن داده‌ها برای هر سفر
                    trip_df = pd.read_csv(trip_csv_path)
                    trip_df.rename(columns={
                                        time_column: 'time',
                                        speed_column: 'Speed',
                                        fuel_column: 'Trip fuel consumption',
                                        mileage_column: 'Cumulative_mileage',
                                        gear_column: 'Current_gear_shift_position_(Current_gear)',
                                        'slope':'slope'
                                    }, inplace=True)
                    
                    # print(trip_df.columns)
                    # پردازش ارتفاعات و شیب
                    

                    st.write(f'Processing and Plotting Results for File {file}...')
                    plot_predicted_vs_real(trip_df, model, model_name ,selected_columns)

                    # حذف فایل موقت سفر
                    

            # حذف فایل‌های موقت
            try:
                os.remove(csv_path)
                os.remove(model_path)
                st.write("Temporary files cleaned up.")
            except Exception as e:
                print(f"Error while cleaning up temporary files: {e}")

            try:
                shutil.rmtree(temp_folder)  # حذف کل پوشه temp
                st.write("Temporary folder cleaned up.")
            except Exception as e:
                print(f"Error while cleaning up temporary folder: {e}")
