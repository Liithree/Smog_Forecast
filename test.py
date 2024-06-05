import json
import pandas as pd

# 从 JSON 文件中读取数据
with open('test.json', 'r') as json_file:
    data = json.load(json_file)
    goalData_Name = "data"
    goalData = data[goalData_Name]

for temp in goalData:
    weather = temp['weather']
    code = weather['code']
    temp['code'] = code
    temp.pop('weather')

csvFileName = 'output.csv'
try:
    # 尝试加载已有的CSV文件
    df = pd.read_csv(csvFileName)
except FileNotFoundError:
    # 如果文件不存在，创建一个新的DataFrame
    df = pd.DataFrame()

df = pd.DataFrame(goalData)
df = pd.concat([df, df], ignore_index=True)

# 将DataFrame保存回CSV文件
df.to_csv(csvFileName, index=False)
print("更新成功！")
