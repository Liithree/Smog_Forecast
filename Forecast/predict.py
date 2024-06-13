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

def load_and_predict():
    api_key = '1ed0fcf8ff944abb86642fd947ed2062'
    api_key0 = 'fb3660c08bce416a989b1d8ee632497c'
    api_key1 = 'c2275ca82b3d4708a13c50f0731ac895'
    api_key2 = '63f243f101a04331a050a707494ace9c'
    api_key3 = '40e0c96902f14b4abd0996c6fb114bd0'
    api_key4 = '5510e07abfaa438ca0e34519dde5bc89'
    api_key5 = 'cf38e99cc37a450db538ea9dbaf68884'
    start_date = '2024-06-12'
    end_date = '2024-06-13'
    lat = 34.34
    lon = 108.94
    url_weather = f'https://api.weatherbit.io/v2.0/history/hourly?lat={lat}&lon={lon}&start_date={start_date}&end_date={end_date}&tz=local&key={api_key1}'
    url_air_quality = f'https://api.weatherbit.io/v2.0/history/airquality?lat={lat}&lon={lon}&start_date={start_date}&end_date={end_date}&tz=local&key={api_key1}'
    response_weather = requests.get(url_weather, stream=True)
    response_air_quality = requests.get(url_air_quality, stream=True)
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
    df_weather = pd.DataFrame(data_weather)
    df_air_quality = pd.DataFrame(data_air_quality)
    df_data = pd.merge(df_weather, df_air_quality, on=['timestamp_local', 'timestamp_utc', 'datetime', 'ts'],
                       how='inner')

    data = df_data
    features = ['rh', 'wind_spd', 'wind_dir', 'vis', 'pres', 'temp']

    # 数据预处理
    scaler = StandardScaler()
    data[features] = scaler.fit_transform(data[features])

    # 生成输入序列
    X_last_24_hours = data[features].values.reshape(1, 24, -1)  # reshape to (1, seq_length, num_features)

    # 转换为torch tensor
    X_last_24_hours_tensor = torch.tensor(X_last_24_hours, dtype=torch.float32)

    model = torch.load('model/model_LSTM_r_complex2.pth')

    # 模型预测
    model.eval()
    with torch.no_grad():
        predicted_aqi = model(X_last_24_hours_tensor).cpu().numpy().flatten()

    # 输出未来24小时的AQI预测值
    print("Predicted AQI for the next 24 hours:")
    for i, aqi in enumerate(predicted_aqi, start=1):
        print(f"Hour {i}: {aqi}")

    # 如果需要将预测值保存到CSV文件
    predicted_aqi_df = pd.DataFrame({'Hour': np.arange(1, len(predicted_aqi) + 1), 'Predicted AQI': predicted_aqi})
    predicted_aqi_df.to_csv('data/predicted_aqi_next_24_hours.csv', index=False)
    print('Predicted AQI values have been saved to predicted_aqi_next_24_hours.csv')






