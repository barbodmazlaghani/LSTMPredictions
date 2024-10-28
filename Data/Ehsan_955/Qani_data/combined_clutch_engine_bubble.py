import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def load_and_combine_data(root_dir):
    combined_data = pd.DataFrame()
    for subdir, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.xlsx'):
                file_path = os.path.join(subdir, file)
                data = pd.read_excel(file_path)
                if 'Engine_speed' in data.columns and 'Clutch_torque' in data.columns:
                    combined_data = pd.concat([combined_data, data[['Engine_speed', 'Clutch_torque']]], ignore_index=True)
    return combined_data

root_dir = r"C:\Users\s_alizadehnia\Desktop\RIRA\Output_Gearshift_Change"
combined_data = load_and_combine_data(root_dir)

# Filter data
combined_data = combined_data[combined_data['Engine_speed'] > 1000]
combined_data = combined_data[combined_data['Clutch_torque'] > 0]

# Define bins for histogram
x_bins = np.arange(0, 6000, 250)
y_bins = np.arange(0, 250, 20)

# Create 2D histogram
hist, xedges, yedges = np.histogram2d(combined_data['Engine_speed'], combined_data['Clutch_torque'], bins=[x_bins, y_bins])
norm_hist = hist / hist.sum()

# List to store points and percentages
points_with_percentage = []

plt.figure(figsize=(10, 6))

for i in range(len(x_bins) - 1):
    for j in range(len(y_bins) - 1):
        in_bin = combined_data[(combined_data['Engine_speed'] >= x_bins[i]) & (combined_data['Engine_speed'] < x_bins[i+1]) &
                               (combined_data['Clutch_torque'] >= y_bins[j]) & (combined_data['Clutch_torque'] < y_bins[j+1])]
        if not in_bin.empty:
            avg_x = in_bin['Engine_speed'].mean()
            avg_y = in_bin['Clutch_torque'].mean()
            percentage = norm_hist[i, j] * 100
            points_with_percentage.append([avg_x, avg_y, percentage])
            plt.scatter(avg_x, avg_y, s=norm_hist[i, j] * 5000, alpha=0.5)  # Size based on bin percentage
            if percentage > 1:
                plt.text(avg_x, avg_y, f"{percentage:.0f}%", ha='center', va='center')

# Sort points by percentage
points_with_percentage_sorted = sorted(points_with_percentage, key=lambda x: x[2], reverse=True)

# Save points and percentages to a CSV file
output_df = pd.DataFrame(points_with_percentage_sorted, columns=['Engine_speed', 'Clutch_torque', 'Percentage'])
output_df.to_csv('C:/Users/s_alizadehnia/Desktop/RIRA/Output_Gearshift_Change/points_with_percentage.csv', index=False)

# Plot settings
plt.xticks(np.arange(0, 6000, 250))
plt.yticks(np.arange(0, 250, 20))
plt.grid(which='major', axis='x', linestyle='-', color='grey', alpha=0.5)
plt.grid(which='major', axis='y', linestyle='-', color='grey', alpha=0.5)
plt.xlim(0, 6000)
plt.ylim(0, 250)

plt.xlabel('Engine Speed')
plt.ylabel('Clutch_torque')
plt.title('Engine Speed vs. Clutch_torque - Average Points in Bins')

plt.show()
