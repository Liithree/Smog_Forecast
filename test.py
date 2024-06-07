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
start_date = '2022-12-01'
end_date = '2023-01-01'
lat = 34.34
lon = 108.94
url_weather = f'https://api.weatherbit.io/v2.0/history/hourly?lat={lat}&lon={lon}&start_date={start_date}&end_date={end_date}&tz=local&key={api_key0}'
url_air_quality = f'https://api.weatherbit.io/v2.0/history/airquality?lat={lat}&lon={lon}&start_date={start_date}&end_date={end_date}&tz=local&key={api_key0}'
response_weather = requests.get(url_weather)
response_air_quality = requests.get(url_air_quality)
print(response_weather.text)
print(response_air_quality.text)
with open('origin_weather_data.txt', 'w') as temp_file1:
    temp_file1.write(response_weather.text)
with open('origin_air_quality_data.txt', 'w') as temp_file2:
    temp_file2.write(response_air_quality.text)
time.sleep(0.1)