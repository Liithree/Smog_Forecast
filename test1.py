import pandas as pd
import numpy as np

# 读取数据
file_name = 'output1.csv'
df = pd.read_csv(file_name)
features = df[
    ['rh', 'wind_spd', 'slp', 'azimuth', 'dewpt', 'snow', 'wind_dir', 'code', 'vis', 'precip', 'pres', 'temp']]
target = df['aqi']

# 检查数据是否存在NaN
print(df.isnull().sum())
print(np.isinf(df).sum())

