import streamlit as st
import pandas as pd
from data_visualizer import display_violin_plots
from file_handler import save_to_npz, handle_uploaded_file


# from data_preprocessor import preprocess_data, normalize_data
# from model_trainer import train_and_save_model, create_bilstm_model
# from tensorflow.keras.models import load_model



st.title("BLSTM Model Trainer and Tester")

# File Upload
st.header("1. Upload CSV/XLSX for Processing")
uploaded_file = st.file_uploader("Upload your file", type=['csv', 'xlsx'])

if uploaded_file:
    # Read the uploaded file
    df = handle_uploaded_file(uploaded_file)
    st.write("Uploaded Data:", df.head())

    # Plot feature distributions
    st.header("2. Feature Distribution with Violin Plots")
    display_violin_plots(df)

    # Preprocess and normalize data
    # st.header("3. Preprocess Data")
    # data, npz_path = preprocess_data(df, uploaded_file.name)
    # st.success(f"Data normalized and saved to {npz_path}.")

    # # Train and save the model
    # st.header("4. Train BiLSTM Model")
    # model_path = train_and_save_model(data, "processed_models")
    # st.success(f"Model trained and saved to {model_path}.")

# Test Model Section
st.header("5. Test the Model")
model_file = st.file_uploader("Upload Model Weights (.h5)", type=['h5'])
test_file = st.file_uploader("Upload Test CSV", type=['csv'])

if model_file and test_file:
    # Load the model
    model = load_model(model_file)
    test_df = pd.read_csv(test_file)
    st.write("Test Data:", test_df.head())

    # Normalize test data
    _, normalized_test_data = normalize_data(test_df.values)
    X_test = normalized_test_data[:, :-1]
    y_test = normalized_test_data[:, -1]

    # Predictions
    predictions = model.predict(X_test)
    st.write("Predictions:", predictions)

    # Visualization
    st.line_chart(predictions.flatten())
    st.write("Actual vs Predicted:")
    result_df = pd.DataFrame({"Actual": y_test.flatten(), "Predicted": predictions.flatten()})
    st.write(result_df)
