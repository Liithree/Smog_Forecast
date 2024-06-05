import json
import pandas as pd


with open('test_air_quality.json', 'r') as f:
    whole_data = json.load(f)
data = whole_data['data']
print(data[0])
print(len(data))
print(data[0].keys())
df = pd.DataFrame(data)

csv_file_path = 'test_air_quality.csv'
df.to_csv(csv_file_path, index=False)



