from datetime import datetime, timedelta

now = datetime.now()
current_hour_timestamp = int(now.replace(minute=0, second=0, microsecond=0).timestamp())
pre_24_hour = now - timedelta(hours=24)
pre_24_hour = pre_24_hour.replace(minute=0, second=0, microsecond=0)
pre_24_hour = int(pre_24_hour.timestamp())
print(current_hour_timestamp)
print(pre_24_hour)