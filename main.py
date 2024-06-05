import requests
import json
import csv


api_key = '1ed0fcf8ff944abb86642fd947ed2062'
start_date = '2024-06-03'
end_date = '2024-06-04'
# latitude 纬度
lat = 34.154
# longitude 经度
lon = 108.563
url = f'https://api.weatherbit.io/v2.0/history/hourly?lat={lat}&lon={lon}&start_date={start_date}&end_date={end_date}&tz=local&key={api_key}'
response = requests.get(url)
data = json.loads(response.text)
print(data)
# 将获取到的data写入到一个json文件中
with open('test.json', 'w') as f:
    json.dump(data, f)


