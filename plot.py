import os
import pandas as pd
import matplotlib.pyplot as plt

def plot_and_save_speed_vs_actual_predicted_dual_y(folder_path, output_folder):
    # Ensure the output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Loop through all CSV files in the folder
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".csv"):
            file_path = os.path.join(folder_path, file_name)

            # Read the CSV file
            data = pd.read_csv(file_path)

            # Check if the file contains the required columns
            if all(col in data.columns for col in ['Actual', 'Predicted']):
                fig, ax1 = plt.subplots(figsize=(10, 6))
                data['Actualm']=data['Actual'].diff()
                data['Predictedm'] = data['Predicted'].diff()
                # Plot Actual and Predicted on the left y-axis
                ax1.set_xlabel('Index')
                ax1.set_ylabel('Actual and Predicted', color='black')
                ax1.plot(data.index, data['Actualm'], label='Actual', color='blue', linestyle='-')
                ax1.plot(data.index, data['Predictedm'], label='Predicted', color='red', linestyle='-')
                ax1.tick_params(axis='y', labelcolor='black')
                ax1.legend(loc='upper left')

                # Create a twin Axes sharing the x-axis for the Speed (right y-axis)
                # ax2 = ax1.twinx()
                # ax2.set_ylabel('Speed', color='green')
                # ax2.plot(data.index, data['Speed'], label='Speed', color='green', linestyle='--')
                # ax2.tick_params(axis='y', labelcolor='green')
                # ax2.legend(loc='upper right')

                # Add a title
                plt.title(f'Speed vs Actual and Predicted - {file_name}')

                # Save the plot
                output_file = os.path.join(output_folder, f"{file_name}_plot.png")
                plt.savefig(output_file)
                plt.close()  # Close the plot to free memory
                print(f"Plot saved to {output_file}")
            else:
                print(f"File {file_name} does not contain the required columns.")

# Specify the folder path where your CSV files are stored
folder_path = r'C:\Users\s_alizadehnia\Desktop\LSTMPredictions\Extended\new_valid_and compare_cool\old_normal_withcool'

# Specify the folder path where you want to save the plots
output_folder = folder_path

# Call the function to generate and save plots for all CSV files
plot_and_save_speed_vs_actual_predicted_dual_y(folder_path, output_folder)
