import os
import pandas as pd
import matplotlib.pyplot as plt


folder_path = r'C:\Users\s_alizadehnia\Desktop\LSTMPredictions\Extended\new_valid_and compare_cool\standard_old_with'
for filename in os.listdir(folder_path):
    old_file_path = os.path.join(folder_path , filename)

    new_file_path = os.path.join(folder_path , f'stan_cool_{filename}')

    os.rename(old_file_path , new_file_path)

#
# import os
# import glob
# import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn.metrics import mean_absolute_percentage_error
#
# def cal_mape(ac,pred):
#     return (abs(ac-pred) / ac).mean
# folder_path = r'C:\Users\s_alizadehnia\Desktop\LSTMPredictions\Extended\compare_standard_normal\With coolant\normal'
# csv_files = glob.glob(os.path.join(folder_path ,'*.csv'))
#
# for file in csv_files:
#     data = pd.read_csv(file)
#
#     ac = data['Actual']
#
#     pred = data['Predicted']
#
#     data['mae'] = (data['Actual'] - data['Predicted'] ).abs()
#     data['mean_mae']=data['mae'].mean()
#     print(data['mae'].mean())
#     output_f = os.path.join(folder_path, os.path.basename(file))
#     data.to_csv(output_f,index=False)