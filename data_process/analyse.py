# 分析每天出现雾霾最多的时段
import pandas as pd
from pyecharts.charts import Bar
from pyecharts import options as opts
from datetime import datetime as dt

# 读取CSV文件
data = pd.read_csv('output1.csv')

# 初始化列表
hour_counts = [0] * 24

data['time'] = pd.to_datetime(data['timestamp_local'])
data['date'] = data['time'].dt.date
data['hour'] = data['time'].dt.hour

hourly_aqi = data.groupby(['date', 'hour'])['aqi'].mean().reset_index()

# 找出每天AQI值最高的4个小时
for date, group in hourly_aqi.groupby('date'):
    top_hours = group.nlargest(4, 'aqi')['hour']
    for hour in top_hours:
        hour_counts[hour] += 1

# 使用pyecharts绘制图像
bar = Bar()
bar.add_xaxis([str(i) + ":00" for i in range(24)])
bar.add_yaxis('次数', hour_counts)
bar.set_global_opts(title_opts=opts.TitleOpts(title='每天AQI值最高的时段'))
bar.render('hourly_aqi.html')
