import pandas as pd
import os
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

def load_and_combine_data(root_dir):
    combined_data = pd.DataFrame()
    for subdir, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.csv'):
                file_path = os.path.join(subdir, file)
                data = pd.read_csv(file_path)
                if 'Engine_speed' in data.columns and 'Clutch_torque' in data.columns:
                    combined_data = pd.concat([combined_data, data[['Engine_speed', 'Clutch_torque']]], ignore_index=True)
    return combined_data

def perform_clustering(data, n_clusters=5):
    # Standardizing the data
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)

    # KMeans clustering
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(data_scaled)

    # Cluster centers in original scale
    centers_original_scale = scaler.inverse_transform(kmeans.cluster_centers_)
    labels = kmeans.labels_

    return centers_original_scale, labels

def plot_clusters(data, centers, labels, n_clusters):
    plt.figure(figsize=(12, 8))
    colors = plt.cm.get_cmap('tab10', n_clusters)

    cluster_counts = pd.Series(labels).value_counts(normalize=True)
    max_count = cluster_counts.max()

    for i in range(n_clusters):
        plt.scatter(data[labels == i]['Engine_speed'], data[labels == i]['Clutch_torque'],
                    color=colors(i), label=f'Cluster {i+1}')

        # Adjusting the circle radius based on the cluster percentage
        cluster_percentage = cluster_counts[i] / max_count
        circle_radius = 2000 * cluster_percentage  # Scale factor for visibility

        plt.scatter(centers[i][0], centers[i][1], color=colors(i), marker='o', edgecolor='black',
                    linewidth=2, s=circle_radius)
        plt.text(centers[i][0], centers[i][1], f'({centers[i][0]:.2f}, {centers[i][1]:.2f}), {cluster_counts[i]*100:.2f}',
                 horizontalalignment='center', verticalalignment='bottom', fontsize=10, color='black')

    plt.xlabel('Engine Speed (Engine_speed)')
    plt.ylabel('Clutch Torque (Clutch_torque)')
    plt.title('KMeans Clustering with Standardization')
    plt.legend()
    plt.grid(True)
    plt.show()

# Main execution
root_dir = r"E:\علم داده\RajabTrips\trips_without_oghab"  # Replace with the path to your directory containing CSV files
combined_data = load_and_combine_data(root_dir)
print(len(combined_data))
centers, labels = perform_clustering(combined_data, n_clusters=5)
plot_clusters(combined_data, centers, labels, n_clusters=5)
