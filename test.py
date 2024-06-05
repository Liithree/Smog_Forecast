# 假设当前天气信息已经全部存储到json文件中，处理它并将有用的信息存储到csv文件中，便于数据处理和操作
import json
import csv

filed_names = ['city_id', 'city_name', 'country_code', ]
with open('test.json', 'r') as f:
    data = json.load(f)

print(data)
print(data['city_id'])



