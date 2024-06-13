# 预测未来24小时的雾霾情况
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import json
import requests
import time
from sklearn.preprocessing import StandardScaler

def clean_weather_data(data):
    for temp in data:
        weather = temp['weather']
        code = weather['code']
        temp['code'] = code
        temp.pop('weather')
    return data


def get_weather():
    api_key = '1ed0fcf8ff944abb86642fd947ed2062'
    api_key0 = 'fb3660c08bce416a989b1d8ee632497c'
    api_key1 = 'c2275ca82b3d4708a13c50f0731ac895'
    api_key2 = '63f243f101a04331a050a707494ace9c'
    api_key3 = '40e0c96902f14b4abd0996c6fb114bd0'
    api_key4 = '5510e07abfaa438ca0e34519dde5bc89'
    api_key5 = 'cf38e99cc37a450db538ea9dbaf68884'
    lat = 34.34
    lon = 108.94
    url_weather = f'https://api.weatherbit.io/v2.0/forecast/hourly?lat={lat}&lon={lon}&key={api_key2}&include=minutely'
    response_weather = requests.get(url_weather, stream=True)
    time.sleep(0.1)
    data_weather = json.loads(response_weather.text)
    print(response_weather.text)
    data_weather = data_weather['data']
    print(data_weather)


get_weather()