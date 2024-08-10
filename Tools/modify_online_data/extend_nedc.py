import pandas as pd


data = pd.read_csv('NEDC_1000_slope_added.csv')

#extra urban cycle
subset = data.loc[814:].copy()

last_time = data['Time'].iloc[-1]
last_fuel_consumption = data['Trip fuel consumption'].iloc[-1]

def adjust_cumulative_fields(df, start_time, start_fuel):
    df['Time'] = df['Time'] - df['Time'].iloc[0] + start_time + 1
    df['Trip fuel consumption'] = df['Trip fuel consumption'] - df['Trip fuel consumption'].iloc[0] + start_fuel
    return df

adjusted_subset1 = adjust_cumulative_fields(subset.copy(), last_time, last_fuel_consumption)
adjusted_subset2 = adjust_cumulative_fields(subset.copy(), adjusted_subset1['Time'].iloc[-1], adjusted_subset1['Trip fuel consumption'].iloc[-1])

final_data = pd.concat([data, adjusted_subset1, adjusted_subset2], ignore_index=True)

final_data.to_csv('Modified_NEDC_1000_slope_added.csv', index=False)
