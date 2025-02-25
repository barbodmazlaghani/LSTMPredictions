import streamlit as st
import pandas as pd
from file_handler import save_to_npz, handle_uploaded_file  # Assuming these are implemented
from data_preprocessor import preprocess_and_save
import os
import streamlit.components.v1 as components

particles_js = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Particles.js</title>
  <style>
  #particles-js {
    position: fixed;
    width: 200vw;
    height: 100vh;
    top: 0;
    left: 0;
    z-index: -1; /* Send the animation to the back */
  }
  .content {
    position: relative;
    z-index: 1;
    color: white;
  }
  
</style>
</head>
<body>
  <div id="particles-js"></div>
  <div class="content">
    <!-- Placeholder for Streamlit content -->
  </div>
  <script src="https://cdn.jsdelivr.net/particles.js/2.0.0/particles.min.js"></script>
  <script>
    particlesJS("particles-js", {
      "particles": {
        "number": {
          "value": 1300,
          "density": {
            "enable": true,
            "value_area": 900
          }
        },
        "color": {
          "value": "#ffffff"
        },
        "shape": {
          "type": "circle",
          "stroke": {
            "width": 0,
            "color": "#000000"
          },
          "polygon": {
            "nb_sides": 5
          },
          "image": {
            "src": "img/github.svg",
            "width": 100,
            "height": 100
          }
        },
        "opacity": {
          "value": 0.5,
          "random": true,
          "anim": {
            "enable": true,
            "speed": 1,
            "opacity_min": 0.2,
            "sync": false
          }
        },
        "size": {
          "value": 2,
          "random": true,
          "anim": {
            "enable": true,
            "speed": 40,
            "size_min": 0.1,
            "sync": false
          }
        },
        "line_linked": {
          "enable": true,
          "distance": 100,
          "color": "#ffffff",
          "opacity": 0.22,
          "width": 1
        },
        "move": {
          "enable": true,
          "speed": 0.8,
          "direction": "none",
          "random": true,
          "straight": false,
          "out_mode": "out",
          "bounce": true,
          "attract": {
            "enable": false,
            "rotateX": 600,
            "rotateY": 1200
          }
        }
      },
      "interactivity": {
        "detect_on": "canvas",
        "events": {
          "onhover": {
            "enable": true,
            "mode": "grab"
          },
          "onclick": {
            "enable": true,
            "mode": "repulse"
          },
          "resize": true
        },
        "modes": {
          "grab": {
            "distance": 100,
            "line_linked": {
              "opacity": 1
            }
          },
          "bubble": {
            "distance": 400,
            "size": 2,
            "duration": 2,
            "opacity": 0.5,
            "speed": 1
          },
          "repulse": {
            "distance": 200,
            "duration": 0.4
          },
          "push": {
            "particles_nb": 2
          },
          "remove": {
            "particles_nb": 3
          }
        }
      },
      "retina_detect": true
    });
  </script>
</body>
</html>
"""


# Add custom CSS to the app
components.html(particles_js,  scrolling=True)


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
    st.write("Uploaded Data:", df.head(5))

    # Initialize session state for tracking steps and column selections
    if "step" not in st.session_state:
        st.session_state.step = 1
        st.session_state.selected_columns = {}
    
    column_options = df.columns.tolist()
    column_options.append(None)
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
            if latitude_column is None:
                print('inja')
                st.session_state.selected_columns["latitude_column"] = None
                st.session_state.step += 1
                st.rerun()
            else:

                st.session_state.selected_columns["latitude_column"] = latitude_column
                st.session_state.step += 1
                st.rerun()


    elif st.session_state.step == 5:
        st.subheader("Step 5: Select the Longitude Column")
        longitude_column = st.selectbox("Select the Longitude Column", options=column_options)
        if st.button("Next"):
            if longitude_column is None:
                st.session_state.selected_columns["longitude_column"] = None
                st.session_state.step += 1
                st.rerun()
            else:
                print(longitude_column)
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
        if st.button("Next"):
            st.session_state.selected_columns["gear_column"] = gear_column
            st.session_state.step += 1
            st.rerun()

    elif st.session_state.step == 9:
        st.subheader("Step 9: Select the Engine speed column")
        engine_speed = st.selectbox("Select the Engine speed column", options=column_options)
        if st.button("Next"):
            st.session_state.selected_columns["engine_speed"] = engine_speed
            st.session_state.step += 1
            st.rerun()

    elif st.session_state.step == 10:
        st.subheader("Step 10: Select the batt voltage Column")
        voltage = st.selectbox("Select the batt voltage Column", options=column_options)
        if st.button("Next"):
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
            selected_columns["coolant_column"],
            selected_columns["mileage_column"],
            selected_columns["gear_column"],
            selected_columns["engine_speed"],
            selected_columns["voltage"]
        ]
        
        # Only add trip column if it exists
        if selected_columns["trip_column"]:
            selected_columns_list.append(selected_columns["trip_column"])
        if selected_columns["latitude_column"]:
            selected_columns_list.append(selected_columns["latitude_column"])
        if selected_columns["longitude_column"]:
            selected_columns_list.append(selected_columns["longitude_column"])
        

        # Create the dataframe with selected columns
        selected_data = df[selected_columns_list]
        print(selected_columns)


        preprocess_and_save(df, selected_columns)
        st.session_state.step += 1

    elif st.session_state.step == 13 or st.session_state.step == 14:

        save_path2 = 'Train.ipynb'
        with open(save_path2, 'rb') as file2:
            st.download_button(label="دانلود فایل IPYNB برای آموزش", data=file2, file_name=save_path2, mime="application/octet-stream")
        
        save_path = 'train_file.npz'

        # بررسی وجود فایل و سپس حذف آن
        if os.path.exists(save_path):
            os.remove(save_path)
            print(f"File {save_path} has been deleted.")
        else:
            print(f"File {save_path} does not exist.")

        st.session_state.step += 1
