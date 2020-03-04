__author__ = "JunSong<songjun@kuaishou.com>"
# Date: 2020/3/3

# 毫秒转日期
import time
timestamp = 1570774556514
time_local = time.localtime(timestamp/1000)
dt = time.strftime("%Y-%m-%d %H:%M:%S", time_local)
print(dt)   # 2019-10-11 14:15:56

# 毫秒转日期
import datetime
d = datetime.datetime.fromtimestamp(timestamp/1000)
# 精确到毫秒
str1 = d.strftime("%Y-%m-%d %H:%M:%S.%f")
print(str1)  # 2019-10-11 14:15:56.514000

# 日期转毫秒
from datetime import datetime
timestr = '2019-01-14 15:22:18.123'
datetime_obj = datetime.strptime(timestr, "%Y-%m-%d %H:%M:%S.%f")
obj_stamp = int(time.mktime(datetime_obj.timetuple()) * 1000.0 + datetime_obj.microsecond / 1000.0)
