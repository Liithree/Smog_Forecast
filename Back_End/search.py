# -*- coding: utf-8 -*-
import json

from django.shortcuts import render

from . import views


# 接收POST请求数据
def search_historyData(request):
    ctx = {}
    if request.POST:
        ctx['rlt'] = request.POST['q']
        # 指定 CSV 文件路径
        csv_file_path = 'cleaned_data.csv'
        # 调用函数提取数据
        data = views.extract_data(csv_file_path)
        # 筛选指定日期的信息
        # 第一行不参与判断
        data = [row for i, row in enumerate(data) if i == 0 or str(ctx['rlt']) in str(row[0])]
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
