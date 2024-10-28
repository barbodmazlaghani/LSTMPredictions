import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def load_and_combine_data(root_dir):
    combined_data = pd.DataFrame()
    for subdir, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.csv'):
                file_path = os.path.join(subdir, file)
                data = pd.read_csv(file_path)
                if 'Engine_speed' in data.columns and 'Accelerator_pedal_position' in data.columns:
                    combined_data = pd.concat([combined_data, data[['Engine_speed', 'Accelerator_pedal_position']]], ignore_index=True)
    return combined_data

root_dir = r"E:\علم داده\RajabTrips\trips_without_oghab"
combined_data = load_and_combine_data(root_dir)
combined_data = combined_data[combined_data['Engine_speed'] > 1000]


x_bins = np.arange(1000, 7001, 500)
y_bins = np.arange(0, 101, 20)

hist, xedges, yedges = np.histogram2d(combined_data['Engine_speed'], combined_data['Accelerator_pedal_position'], bins=[x_bins, y_bins])
norm_hist = hist / hist.sum()

plt.figure(figsize=(10, 6))

for i in range(len(x_bins) - 1):
    for j in range(len(y_bins) - 1):
        in_bin = combined_data[(combined_data['Engine_speed'] >= x_bins[i]) & (combined_data['Engine_speed'] < x_bins[i+1]) &
                               (combined_data['Accelerator_pedal_position'] >= y_bins[j]) & (combined_data['Accelerator_pedal_position'] < y_bins[j+1])]
        if not in_bin.empty:
            avg_x = in_bin['Engine_speed'].mean()
            avg_y = in_bin['Accelerator_pedal_position'].mean()
            plt.scatter(avg_x, avg_y, s=norm_hist[i, j] * 5000, alpha=0.5)  # Size based on bin percentage
            if norm_hist[i, j]*100 > 1:
                plt.text(avg_x, avg_y, f"{norm_hist[i, j]*100:.0f}%", ha='center', va='center')

plt.xticks(np.arange(1000, 7001, 500))
plt.yticks(np.arange(0, 101, 20))
plt.grid(which='major', axis='x', linestyle='-', color='grey', alpha=0.5)
plt.grid(which='major', axis='y', linestyle='-', color='grey', alpha=0.5)
plt.xlim(1000, 7000)
plt.ylim(0, 100)

plt.xlabel('Engine Speed')
plt.ylabel('Accelerator Pedal Position')
plt.title('Engine Speed vs. Accelerator Pedal Position - Average Points in Bins')

plt.show()
