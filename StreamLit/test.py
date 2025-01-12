import streamlit as st
import pandas as pd
from file_handler import save_to_npz, handle_uploaded_file  # Assuming these are implemented
from data_preprocessor import preprocess_and_save


@st.cache_data
def read_file(uploaded_file):
    """Handle file upload and caching."""
    return handle_uploaded_file(uploaded_file)


# Streamlit Title
st.title("BLSTM Model Trainer and Tester")

# File Upload
st.header("1. Upload CSV/XLSX for Processing")
uploaded_file = st.file_uploader("Upload your file", type=['csv', 'xlsx'])

if uploaded_file:
    # Read the uploaded file
    df = read_file(uploaded_file)
    st.write("Uploaded Data:", df.head(10))

    # Initialize session state for tracking steps and column selections
    if "step" not in st.session_state:
        st.session_state.step = 1
        st.session_state.selected_columns = {}
    
    column_options = df.columns.tolist()
    # Step-by-step column selection
    if st.session_state.step == 1:
        st.subheader("Step 1: Select the Time Column")
        time_column = st.selectbox("Select the Time Column", options=column_options)
        if st.button("Next"):
            st.session_state.selected_columns["time_column"] = time_column
            st.session_state.step += 1
            st.rerun()

    elif st.session_state.step == 2:
        st.subheader("Step 2: Select the Speed Column")
        speed_column = st.selectbox("Select the Speed Column", options=column_options)
        if st.button("Next"):
            st.session_state.selected_columns["speed_column"] = speed_column
            st.session_state.step += 1
            st.rerun()


    elif st.session_state.step == 3:
        st.subheader("Step 3: Select the Trip Fuel Consumption Column")
        fuel_column = st.selectbox("Select the Trip Fuel Consumption Column", options=column_options)
        if st.button("Next"):
            st.session_state.selected_columns["fuel_column"] = fuel_column
            st.session_state.step += 1
            st.rerun()


    elif st.session_state.step == 4:
        st.subheader("Step 4: Select the Latitude Column")
        latitude_column = st.selectbox("Select the Latitude Column", options=column_options)
        if st.button("Next"):
            st.session_state.selected_columns["latitude_column"] = latitude_column
            st.session_state.step += 1
            st.rerun()


    elif st.session_state.step == 5:
        st.subheader("Step 5: Select the Longitude Column")
        longitude_column = st.selectbox("Select the Longitude Column", options=column_options)
        if st.button("Next"):
            st.session_state.selected_columns["longitude_column"] = longitude_column
            st.session_state.step += 1
            st.rerun()


    elif st.session_state.step == 6:
        st.subheader("Step 6: Select the Coolant Temperature Column")
        coolant_column = st.selectbox("Select the Coolant Temperature Column", options=column_options)
        if st.button("Next"):
            st.session_state.selected_columns["coolant_column"] = coolant_column
            st.session_state.step += 1
            st.rerun()


    elif st.session_state.step == 7:
        st.subheader("Step 7: Select the Cumulative Mileage Column")
        mileage_column = st.selectbox("Select the Cumulative Mileage Column", options=column_options)
        if st.button("Next"):
            st.session_state.selected_columns["mileage_column"] = mileage_column
            st.session_state.step += 1
            st.rerun()


    elif st.session_state.step == 8:
        st.subheader("Step 8: Select the Current Gear Column")
        gear_column = st.selectbox("Select the Current Gear Column", options=column_options)
        if st.button("Finish"):
            st.session_state.selected_columns["gear_column"] = gear_column
            st.session_state.step += 1
            st.rerun()

    elif st.session_state.step == 9:
        st.subheader("Step 9: Select the Engine speed column")
        engine_speed = st.selectbox("Select the Engine speed column", options=column_options)
        if st.button("Finish"):
            st.session_state.selected_columns["engine_speed"] = engine_speed
            st.session_state.step += 1
            st.rerun()

    elif st.session_state.step == 10:
        st.subheader("Step 10: Select the batt voltage Column")
        voltage = st.selectbox("Select the batt voltage Column", options=column_options)
        if st.button("Finish"):
            st.session_state.selected_columns["voltage"] = voltage
            st.session_state.step += 1
            st.rerun()


    # Step 9: Ask about trip column (new step)
    elif st.session_state.step == 11:
        st.subheader("Step 11: Does your CSV have a Trip Column?")
        trip_answer = st.radio("Does the CSV contain a Trip column?", options=["Yes", "No"])

        if trip_answer == "Yes":
            st.subheader("Select the Trip Column")
            trip_column = st.selectbox("Select the Trip Column", options=column_options)
            if st.button("Next"):
                st.session_state.selected_columns["trip_column"] = trip_column
                st.session_state.step += 1
                st.rerun()


        elif trip_answer == "No":
            # If no trip column, set gear_column to None
            st.session_state.selected_columns["trip_column"] = None
            st.session_state.step += 1
            st.rerun()


    # Final Step: Confirm and Save Data
    elif st.session_state.step == 12:
        st.subheader("Step 12: Confirm Your Selections")
        selected_columns = st.session_state.selected_columns
        st.write("Your Selections:")
        st.write("Time Column:", selected_columns["time_column"])
        st.write("Speed Column:", selected_columns["speed_column"])
        st.write("Fuel Column:", selected_columns["fuel_column"])
        st.write("Latitude Column:", selected_columns["latitude_column"])
        st.write("Longitude Column:", selected_columns["longitude_column"])
        st.write("Coolant Column:", selected_columns["coolant_column"])
        st.write("Mileage Column:", selected_columns["mileage_column"])
        st.write("Gear Column:", selected_columns["gear_column"])
        st.write("engine_speed:", selected_columns["engine_speed"])
        st.write("batt voltage:", selected_columns["voltage"])

        if selected_columns["trip_column"]:
            st.write("Trip Column:", selected_columns["trip_column"])
        else:
            st.write("No Trip Column selected.")

        # Combine selected columns
        selected_columns_list = [
            selected_columns["time_column"],
            selected_columns["speed_column"],
            selected_columns["fuel_column"],
            selected_columns["latitude_column"],
            selected_columns["longitude_column"],
            selected_columns["coolant_column"],
            selected_columns["mileage_column"],
            selected_columns["gear_column"],
            selected_columns["engine_speed"],
            selected_columns["voltage"]
        ]
        
        # Only add trip column if it exists
        if selected_columns["trip_column"]:
            selected_columns_list.append(selected_columns["trip_column"])

        # Create the dataframe with selected columns
        selected_data = df[selected_columns_list]
        print(selected_columns)


        preprocess_and_save(df, selected_columns)
        st.session_state.step += 1
