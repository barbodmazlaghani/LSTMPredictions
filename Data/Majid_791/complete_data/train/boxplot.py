import os
import pandas as pd
import matplotlib.pyplot as plt

file_path = 'data_report7_train.xlsx'
output_dir = 'boxplots'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

df = pd.read_excel(file_path)

for column in df.select_dtypes(exclude=['object']).columns:
    plt.figure()
    df.boxplot(column=column)
    plt.title(f'Boxplot of {column}')
    plt.savefig(os.path.join(output_dir, f'{column}_boxplot.png'))
    plt.close()

print(f'Boxplots have been saved in the directory: {output_dir}')
