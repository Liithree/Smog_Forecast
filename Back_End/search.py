# -*- coding: utf-8 -*-
import json

from django.shortcuts import render

from . import views, predict


# 接收POST请求数据
def search_historyData(request):
    ctx = {}
    if request.POST:
        ctx['rlt'] = request.POST['q']
        date_str = str(ctx['rlt'])
        data_str = str(ctx['rlt'])

        # 使用切片替换字符串中的短横线
        data_str = data_str.replace('-', '/')
        data_str = "/".join(part.lstrip('0') for part in data_str.split('/'))
        ds = data_str + ' '
        # 指定 CSV 文件路径
        csv_file_path = 'cleaned_data.csv'
        # 调用函数提取数据
        data = views.extract_data(csv_file_path)
        # 筛选指定日期的信息
        # 第一行不参与判断
        data = [row for i, row in enumerate(data) if i == 0 or ds in str(row[0])]
        for item in data:
            for _ in range(27):
                item.pop(1)
            for _ in range(10):
                item.pop(2)
        # 将数据转换为 JSON 格式
        data_json = json.dumps(data)
        print(data_json)
        return render(request, "historyDatapage.html", {"rlt": data_json})
    else:
        return render(request, "historyDatapage.html", {"rlt": ctx})


#预测数据
def search_ForecastData(request):
    ctx = {}
    if request.POST:
        #调用预测方法；
        data = predict.get_weather()
        ctx['rlt'] = request.POST['q']

        # 将数据转换为 JSON 格式
        data_json = json.dumps(data)

        return render(request, "forecastData.html", {"rlt": data_json})
    else:
        return render(request, "forecastData.html", {"rlt": ctx})
