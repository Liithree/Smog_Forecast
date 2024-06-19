import pandas as pd
from datetime import datetime


def extract_date_info(timestamp):
    dt = datetime.strptime(str(timestamp), "%Y-%m-%d %H:%M:%S")
    return dt.month, dt.day, dt.hour


def classify_aqi(aqi):
    if aqi <= 50:
        return 0
    elif aqi <= 100:
        return 1
    elif aqi <= 150:
        return 2
    elif aqi <= 200:
        return 3
    elif aqi <= 300:
        return 4
    else:
        return 5


df = pd.read_csv('../output.csv')
df['timestamp_local'] = pd.to_datetime(df['timestamp_local'])
df = df.drop_duplicates(subset=['timestamp_local'])
df = df.sort_values(by='timestamp_local')
df = df.reset_index(drop=True)
df = df.dropna(subset='aqi')
df[['month', 'day', 'hour']] = df['timestamp_local'].apply(lambda x: pd.Series(extract_date_info(x)))
df['aqi_level'] = df['aqi'].apply(classify_aqi)
# 将‘ts’放在第一列
cols = ['timestamp_local'] + [col for col in df.columns if col != 'timestamp_local']
df = df[cols]
df.to_csv('cleaned_data.csv', index=False)

