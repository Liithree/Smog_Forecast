import json
import os
from django.http import JsonResponse
from django.shortcuts import render
import pandas as pd
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
