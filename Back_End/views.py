import json
import os
from django.http import JsonResponse
from django.shortcuts import render
import pandas as pd
from .current_weather import get_weather
import csv

def extract_data(csv_file_path):
    data = []
    with open(csv_file_path, 'r') as csv_file:
        csv_reader = csv.reader(csv_file)
        headers = next(csv_reader)
        data.append(headers)
        for row in csv_reader:
            data.append(row)
    return data

def homepage(request):

    # 调用get_weather将当前天气情况写入到current_weather_data的两个json文件中
    get_weather()
    # 读取json文件
    json_file_path = os.path.join('current_weather_data', 'data.json') # 名字要改
    try:
        with open(json_file_path, 'r', encoding='utf-8') as file:
            # 从文件中加载 JSON 数据
            json_data = json.load(file)
    except Exception as e:
        json_data = None
        error_message = str(e)
        return render(request, '1.html', {'error': error_message})

    # 将数据传递给模板
    return render(request, 'xx.html', {'data': json_data})

    # 指定 CSV 文件路径
    csv_file_path = 'cleaned_data.csv'
    # 调用函数提取数据
    data = extract_data(csv_file_path)

    for item in data:
        for _ in range(27):
            item.pop(1)
        for _ in range(10):
            item.pop(2)
    # 将数据转换为 JSON 格式
    data_json = json.dumps(data)
    return render(request, "homepage.html", {"names": data_json})




