# 预测未来24小时的雾霾情况
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import json
import requests
from datetime import datetime, timedelta
import time
from sklearn.preprocessing import StandardScaler
pd.options.mode.chained_assignment = None


# 定义更加复杂的LSTM模型用于回归
class AirQualityLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout=0.5):
        super(AirQualityLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout, bidirectional=True)
        self.fc1 = nn.Linear(hidden_size * 2, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, 1)
        self.dropout = nn.Dropout(dropout)
        self.batch_norm1 = nn.BatchNorm1d(hidden_size)
        self.batch_norm2 = nn.BatchNorm1d(hidden_size // 2)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size)
        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :]
        out = self.fc1(out)
        out = self.batch_norm1(out)
        out = torch.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.batch_norm2(out)
        out = torch.relu(out)
        out = self.dropout(out)
        out = self.fc3(out)
        return out


def clean_weather_data(data):
    for temp in data:
        weather = temp['weather']
        code = weather['code']
        temp['code'] = code
        temp.pop('weather')
    return data


def get_time():
    now = datetime.now()
    current_hour_timestamp = int(now.replace(minute=0, second=0, microsecond=0).timestamp())
    pre_24_hour = now - timedelta(hours=24)
    pre_24_hour = pre_24_hour.replace(minute=0, second=0, microsecond=0)

    current_hour_formatted = now.strftime('%Y-%m-%d:%H')
    pre_24_hour_formatted = pre_24_hour.strftime('%Y-%m-%d:%H')
    return pre_24_hour_formatted, current_hour_formatted


def get_time_2():
    now = datetime.now()
    current_hour_timestamp = int(now.replace(minute=0, second=0, microsecond=0).timestamp())
    pre_24_hour = now - timedelta(hours=24)
    pre_24_hour = pre_24_hour.replace(minute=0, second=0, microsecond=0)
    pre_24_hour = int(pre_24_hour.timestamp())
    return pre_24_hour, current_hour_timestamp


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
    start_date, end_date = get_time()
    url_pre_weather = f'https://api.weatherbit.io/v2.0/history/hourly?lat={lat}&lon={lon}&start_date={start_date}&end_date={end_date}&tz=local&key={api_key3}'
    url_weather = f'https://api.weatherbit.io/v2.0/forecast/hourly?lat={lat}&lon={lon}&key={api_key3}&include=minutely'
    response_pre_weather = requests.get(url_pre_weather, stream=True)
    response_weather = requests.get(url_weather, stream=True)
    time.sleep(0.1)
    data_pre_weather = json.loads(response_pre_weather.text)
    data_weather = json.loads(response_weather.text)
    # print(data_pre_weather)
    # print('=================')
    # print(data_weather)
    data_pre_weather = data_pre_weather['data']
    data_weather = data_weather['data']
    # print(data_weather)
    data_pre_weather = clean_weather_data(data_pre_weather)
    data_weather = clean_weather_data(data_weather)
    df_pre_weather = pd.DataFrame(data_pre_weather)
    df_weather = pd.DataFrame(data_weather)
    # print(df_weather)
    data = df_pre_weather._append(df_weather, ignore_index=True)
    data = data.sort_values(by='ts')
    # data.to_csv('temp.csv', index=False)
    features = ['rh', 'wind_spd', 'wind_dir', 'vis', 'pres', 'temp']

    start, end = get_time_2()
    i = 0
    my_list = []
    while i < 24:
        data_temp = data[(data['ts'] >= start) & (data['ts'] < end)]
        # print(data_temp)
        # print('=============')
        # 数据预处理
        scaler = StandardScaler()
        data_temp[features] = scaler.fit_transform(data_temp[features])

        # 生成输入序列
        X_last_24_hours = data_temp[features].values.reshape(1, 24, -1)  # reshape to (1, seq_length, num_features)

        # 转换为torch tensor
        X_last_24_hours_tensor = torch.tensor(X_last_24_hours, dtype=torch.float32)

        # 模型路径
        model = torch.load('model_LSTM_r_complex2.pth')

        # 模型预测
        model.eval()
        with torch.no_grad():
            predicted_aqi = model(X_last_24_hours_tensor).cpu().numpy().flatten()

        # 输出未来24小时的AQI预测值
        my_list.append([i, predicted_aqi[0]])
        print(i, predicted_aqi[0])

        # 如果需要将预测值保存到CSV文件
        # predicted_aqi_df = pd.DataFrame({'Hour': np.arange(1, len(predicted_aqi) + 1), 'Predicted AQI': predicted_aqi})
        # predicted_aqi_df.to_csv('data/predicted_aqi_next_24_hours.csv', index=False)
        # print('Predicted AQI values have been saved to predicted_aqi_next_24_hours.csv')
        start += 3600
        end += 3600
        i += 1
    return my_list

print(get_weather())
# get_weather()
