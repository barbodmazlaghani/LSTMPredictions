import os
import pandas as pd
import matplotlib.pyplot as plt
import glob

# Change directory to where the script is located
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Get a list of all CSV files in the directory
csv_files = glob.glob("*.csv")

# Iterate over each CSV file
for csv_file in csv_files:
    df = pd.read_csv(csv_file)

    # Check if the column 'Coolant_temperature' exists
    if 'Coolant_temperature' in df.columns:
        # Plotting the 'Coolant_temperature' column
        plt.figure(figsize=(10, 6))
        plt.plot(df['Coolant_temperature'])
        plt.title(f'Coolant Temperature Over Time - {csv_file}')
        plt.xlabel('Index')
        plt.ylabel('Coolant Temperature')
        plt.grid(True)

        # Save the figure as a PNG file with a filename based on the CSV file name
        fig_name = f'{os.path.splitext(csv_file)[0]}_coolant_temperature.png'
        plt.savefig(fig_name)
        plt.close()  # Close the figure to free up memory

        print(f"Figure saved as {fig_name}")
    else:
        print(f"'Coolant_temperature' column not found in {csv_file}.")
