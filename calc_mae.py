import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_percentage_error

def cal_mape(ac,pred):
    return (abs(ac-pred) / ac).mean
folder_path = r'C:\Users\s_alizadehnia\Desktop\LSTMPredictions\Extended\new_valid_and compare_cool\normal_old_without'
txt_file = f'{folder_path}/out.txt'
csv_files = glob.glob(os.path.join(folder_path ,'*.csv'))

for files in csv_files:
    data = pd.read_csv(files)

    ac = data['Actual']
    print(files)
    pred = data['Predicted']

    data['mae'] = ((data['Actual'].iloc[1:] - data['Predicted'].iloc[1:] ).abs()) / data['Actual'].iloc[1:]
    data['mean_mae']=(data['mae'].mean()) * 100
    data['last_error'] = (abs(data['Actual'].iloc[-1] - data['Predicted'].iloc[-1]) / data['Actual'].iloc[-1]) *100
    output_f = os.path.join(folder_path, os.path.basename(files))
    a = (os.path.basename(files))
    data.to_csv(output_f,index=False)
    value= {'mae' : data['mean_mae'].iloc[-1] , 'last_er': data['last_error'].iloc[-1]}
    with open(txt_file , mode='a') as file :
        file.write(f'{a}_____{value}\n')

