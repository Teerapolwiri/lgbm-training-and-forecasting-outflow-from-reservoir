#%%
import lightgbm as lgb
import numpy as np
import pandas as pd

loaded_gbm = lgb.Booster(model_file= r'Model\lgb_model_200102.txt')

# [Inflow_Stage1, Current_Storage, Outflow_Lag1]
df = pd.read_csv(r'data\input_pmp.txt')

current_reservoir_storage = df.loc[0, 'Volume mcm']
yesterday_outflow = df.loc[0, 'Qout cms']

# Loop through each day in the data
for i in range(len(df)):
    today_inflow_forecast = df.loc[i, 'Qin cms']
    date = df.loc[i, 'date']
    
    X_new_condition = np.array([[today_inflow_forecast, current_reservoir_storage, yesterday_outflow]])
    predicted_outflow = loaded_gbm.predict(X_new_condition)
    raw_outflow_value = predicted_outflow[0]
    
    print(f"วันที่ {date}: Inflow: {today_inflow_forecast}, Volume: {current_reservoir_storage:.2f}, Predicted Outflow: {raw_outflow_value:.2f} m^3/s")
    
    # Update volume and yesterday_outflow for the next day
    # Convert units: qin/qout in cms, volume in mcm
    # delta_volume (mcm) = (qin - qout) * 86400 seconds / 1,000,000 = (qin - qout) * 0.0864
    current_reservoir_storage += (today_inflow_forecast - raw_outflow_value) * 0.0864
    yesterday_outflow = raw_outflow_value
# %%
