import os
import pandas
import pandas as pd

path = 'C:/Users/s_alizadehnia/Desktop/LSTMPredictions/Data/Ehsan_955/slope_added/train/'
# Col = Vehicle_Speed
# min = 0
# max = 119
# mean = 46.16431309259633
# median = 50.0
# std = 30.365752353940135
#
#
# Col = Acceleration
# min = -7.0
# max = 10.0
# mean = -0.007817228162194876
# median = 0.0
# std = 1.0594805487209455
#
#
# Col = Momentary fuel consumption
# min = -1362.375
# max = 6625.875
# mean = 472.4091618418398
# median = 262.875
# std = 603.0249535728824
#
#
# Col = slope
# min = -6.626506024089705
# max = 11.62790697658875
# mean = 0.02705546081838114
# median = 0.0
# std = 2.0500977589271807
#


path = 'C:/Users/s_alizadehnia/Desktop/LSTMPredictions/Data/Majid_791/complete_data/train'
# Col = Vehicle_Speed
# min = 0
# max = 125
# mean = 59.25432447710571
# median = 69.0
# std = 34.279712568434974
#
#
# Col = Acceleration
# min = -17.0
# max = 10.0
# mean = -0.006387789711701526
# median = 0.0
# std = 1.7528358773616612
#
#
# Col = Momentary fuel consumption
# min = -1576.8120000000345
# max = 11882.25
# mean = 1294.281350706614
# median = 892.8360000000102
# std = 1357.51158871322
#
#
# Col = slope
# min = -5.829596412557195
# max = 4.299065420561917
# mean = -0.001803071032386209
# median = 0.0
# std = 1.355041906920822



all_data = []

for filename in os.listdir(path):
    if filename.endswith('.csv'):
        file_path = os.path.join(path , filename)
        df = pd.read_csv(file_path)
        df['Momentary fuel consumption'] = df['Trip_fuel_consumption'].diff().fillna(0)
        df['Acceleration'] = df['Vehicle_Speed'].diff().fillna(0)
        all_data.append(df)


combined_df = pd.concat(all_data , ignore_index=True)




selects = ['Vehicle_Speed','Acceleration','Momentary fuel consumption','slope']
for select in selects:
    min = combined_df[select].min()
    max = combined_df[select].max()
    mean = combined_df[select].mean()
    median = combined_df[select].median()
    std = combined_df[select].std()
    print(f'Col = {select}')
    print(f'min = {min}')
    print(f'max = {max}')
    print(f'mean = {mean}')
    print(f'median = {median}')
    print(f'std = {std} \n\n')
