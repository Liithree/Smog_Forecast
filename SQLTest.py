# -*- coding:utf-8 -*-
"""
作者：Lenovo
日期：2024年06月07日
"""

import pymysql
db = pymysql.connect(host='localhost',
                     user='root',
                     password='123456',
                     database='city_lat_lon',
                     charset='utf8')

# 使用 cursor() 方法创建一个游标对象 cursor
cursor = db.cursor()

city_name = input("您想查找的城市为：")

sql = "SELECT latitude, longitude FROM city_lat_lon WHERE name = '%s'" % city_name

cursor.execute(sql)
result = cursor.fetchall()
for col in result:
    latitude = col[0]
    longitude = col[1]
    print("纬度为%f,经度为%f" %(latitude,longitude))
# 使用 execute()  方法执行 SQL 查询
#cursor.execute("SELECT VERSION()")

# 使用 fetchone() 方法获取单条数据.
#data = cursor.fetchone()

#print("数据库连接成功！")

# 关闭数据库连接
db.close()
