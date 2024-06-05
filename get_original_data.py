import requests
import json

# 要获取天气信息的城市名称
city_name = "Shanghai"

# 用于访问OpenWeatherMap API的API密钥
api_key = "d2988389f8195733705c581475d92106"

# 使用API密钥和城市名称构建API URL
url = f"https://api.openweathermap.org/data/2.5/weather?q={city_name}&appid={api_key}&units=metric"

# 发送请求并获取响应
response = requests.get(url)

# 解析响应中的JSON数据
data = json.loads(response.text)

print('====================')
print(data)
print('====================')
# 提取所需的天气信息
temperature = data["main"]["temp"]
humidity = data["main"]["humidity"]
description = data["weather"][0]["description"]


# 打印天气信息
print(f"The temperature in {city_name} is {temperature} Celsius.")
print(f"The humidity in {city_name} is {humidity}%.")
print(f"The weather description in {city_name} is {description}.")


# Result:
# The temperature in Shanghai is 302.95 Kelvin.
# The humidity in Shanghai is 45%.
# The weather description in Shanghai is scattered clouds.
