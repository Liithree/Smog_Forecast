import json
import requests
import pandas as pd
import time
from pymysql import Connection

def clean_weather_data(data):
    for temp in data:
        weather = temp['weather']
        code = weather['code']
        temp['code'] = code
        temp.pop('weather')
    return data


api_key = '1ed0fcf8ff944abb86642fd947ed2062'
api_key0 = 'fb3660c08bce416a989b1d8ee632497c'
api_key1 = 'c2275ca82b3d4708a13c50f0731ac895'
api_key2 = '63f243f101a04331a050a707494ace9c'
api_key3 = '40e0c96902f14b4abd0996c6fb114bd0'
api_key4 = '5510e07abfaa438ca0e34519dde5bc89'
api_key5 = 'cf38e99cc37a450db538ea9dbaf68884'
start_date = '2022-04-01'
end_date = '2022-05-01'
lat = 34.34
lon = 108.94
url_weather = f'https://api.weatherbit.io/v2.0/history/hourly?lat={lat}&lon={lon}&start_date={start_date}&end_date={end_date}&tz=local&key={api_key1}'
url_air_quality = f'https://api.weatherbit.io/v2.0/history/airquality?lat={lat}&lon={lon}&start_date={start_date}&end_date={end_date}&tz=local&key={api_key1}'
response_weather = requests.get(url_weather, stream=True)
response_air_quality = requests.get(url_air_quality, stream=True)
print(response_weather.text)
print(response_air_quality.text)
with open('origin_weather_data.txt', 'w') as temp_file1:
    temp_file1.write(response_weather.text)
with open('origin_air_quality_data.txt', 'w') as temp_file2:
    temp_file2.write(response_air_quality.text)
time.sleep(0.1)
data_weather = json.loads(response_weather.text)
# with open('weather_data.json', 'w') as weather_data_file:
#     json.dump(data_weather, weather_data_file)
data_air_quality = json.loads(response_air_quality.text)
# with open('air_quality_data.json', 'w') as air_quality_file:
#     json.dump(data_air_quality, air_quality_file)
# print(data_air_quality)
# 获取目标数据
data_weather = data_weather['data']
data_air_quality = data_air_quality['data']

# 筛选天气数据
data_weather = clean_weather_data(data_weather)

file_name = 'output.csv'
df = pd.read_csv(file_name)
df_weather = pd.DataFrame(data_weather)
df_air_quality = pd.DataFrame(data_air_quality)
# print(df_weather.info())
# print(df_air_quality.info())
df_data = pd.merge(df_weather, df_air_quality, on=['timestamp_local', 'timestamp_utc', 'datetime', 'ts'], how='inner')
df = pd.concat([df, df_data], ignore_index=True)
df.to_csv(file_name, index=False)

