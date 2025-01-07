import os
import pandas as pd
import matplotlib.pyplot as plt

def plot_and_save_speed_vs_actual_predicted_dual_y(root_folder):
    # Walk through the directory tree to find all CSV files
    for dirpath, _, filenames in os.walk(root_folder):
        for file_name in filenames:
            if file_name.endswith(".csv"):
                file_path = os.path.join(dirpath, file_name)

                # Read the CSV file
                data = pd.read_csv(file_path)

                # Check if the file contains the required columns
                if all(col in data.columns for col in ['Actual', 'Predicted','rate']):
                    if data['rate'].notna().any():
                        fig, ax1 = plt.subplots(figsize=(10, 6))
                        data['Actualm'] = data['rate']
                        data['Predictedm'] = data['Predicted']

                        # Plot Actual and Predicted on the left y-axis
                        ax1.set_xlabel('Index')
                        ax1.set_ylabel('Actual and Predicted', color='black')
                        ax1.plot(data.index, data['Actualm'], label='Actual', color='blue', linestyle='-')
                        ax1.plot(data.index, data['Predictedm'], label='Predicted', color='red', linestyle='-')
                        ax1.tick_params(axis='y', labelcolor='black')
                        ax1.legend(loc='upper left')

                        # Add a title
                        plt.title(f'Speed vs Actual and Predicted - {file_name}')

                        # Save the plot next to the CSV file
                        output_file = os.path.join(dirpath, f"{os.path.splitext(file_name)[0]}_plot.png")
                        plt.savefig(output_file)
                        plt.close()  # Close the plot to free memory
                        print(f"Plot saved to {output_file}")
                    else:
                        fig, ax1 = plt.subplots(figsize=(10, 6))
                        data['Actualm'] = data['Actual'].diff()
                        data['Predictedm'] = data['Predicted']*1000

                        # Plot Actual and Predicted on the left y-axis
                        ax1.set_xlabel('Index')
                        ax1.set_ylabel('Actual and Predicted', color='black')
                        ax1.plot(data.index, data['Actualm'], label='Actual', color='blue', linestyle='-')
                        ax1.plot(data.index, data['Predictedm'], label='Predicted', color='red', linestyle='-')
                        ax1.tick_params(axis='y', labelcolor='black')
                        ax1.legend(loc='upper left')

                        # Add a title
                        plt.title(f'Speed vs Actual and Predicted - {file_name}')

                        # Save the plot next to the CSV file
                        output_file = os.path.join(dirpath, f"{os.path.splitext(file_name)[0]}_plot.png")
                        plt.savefig(output_file)
                        plt.close()  # Close the plot to free memory
                        print(f"Plot saved to {output_file}")
                else:
                    print(f"File {file_name} does not contain the required columns.")

# Specify the root folder path where your CSV files are stored
root_folder = r'C:\Users\s_alizadehnia\Desktop\LSTMPredictions\Extended\513_data_test1\tes for cold and hot trip\513_model_2slices_tripcons'

# Call the function to generate and save plots for all CSV files
plot_and_save_speed_vs_actual_predicted_dual_y(root_folder)


