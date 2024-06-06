import json
import requests
import pandas as pd
from pymysql import Connection

def clean_weather_data(data):
    for temp in data:
        weather = temp['weather']
        code = weather['code']
        temp['code'] = code
        temp.pop('weather')
    return data


api_key = '1ed0fcf8ff944abb86642fd947ed2062'
start_date = '2024-06-03'
end_date = '2024-06-04'
lat = 34.108626
lon = 108.605011
url_weather = f'https://api.weatherbit.io/v2.0/history/hourly?lat={lat}&lon={lon}&start_date={start_date}&end_date={end_date}&tz=local&key={api_key}'
url_air_quality = f'https://api.weatherbit.io/v2.0/history/airquality?lat={lat}&lon={lon}&start_date={start_date}&end_date={end_date}&tz=local&key={api_key}'
response_weather = requests.get(url_weather)
response_air_quality = requests.get(url_air_quality)
data_weather = json.loads(response_weather.text)
data_air_quality = json.loads(response_air_quality.text)

# 获取目标数据
data_weather = data_weather['data']
data_air_quality = data_air_quality['data']

# 筛选天气数据
data_weather = clean_weather_data(data_weather)

file_name = 'output.csv'

df_weather = pd.DataFrame(data_weather)
df_air_quality = pd.DataFrame(data_air_quality)
# print(df_weather.info())
# print(df_air_quality.info())
df_data = pd.merge(df_weather, df_air_quality, on=['timestamp_local', 'timestamp_utc', 'datetime', 'ts'], how='inner')
print(df_data.info())
print(df_data)
df_data.to_csv(file_name, index=False)

