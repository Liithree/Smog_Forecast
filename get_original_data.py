from collections.abc import MutableMapping
import requests
import json
from bs4 import BeautifulSoup

# 获取数据
# OpenSky Network API endpoint
url = "https://opensky-network.org/api/states/all"

# Define parameters for a specific aircraft or flight (e.g., icao24 code)

response = requests.get(url)
data = response.json()

print(json.dumps(data, indent=4))
