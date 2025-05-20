import re
from datetime import datetime
import pytz

txt = "20250410  13:43:15    Asia/Tokyo"
# txt = "20250410  13:43:15    Asia/Tddsf"
p = re.compile(r"(\d+)\s+(\d\d:\d\d:\d\d)\s+(\S+)")
x = p.match(txt)
print(x.group(0))

# print(x.group(1))
dt = x.group(1)
print(f"dt = {dt}")

# print(x.group(2))
hms = x.group(2)
print(f"hms = {hms}")

# print(x.group(3))
tzone = x.group(3)
print(f"tzone = {tzone}")

dt_dt = datetime.strptime(f"{dt} {hms}", "%Y%m%d %H:%M:%S")
print(f"dt_dt = {dt_dt}")

local_tz = pytz.timezone(tzone)

localtime = dt_dt.astimezone(pytz.timezone(tzone))
print(f"localtime = {localtime}")

timestamp = int(localtime.timestamp())
print(f"timestamp = {timestamp}")

jpt = '20250410 13:43:15 Asia/Tokyo'
dateTime = jpt[0:17]
timeZone = jpt[18:]
jptsp = 1744260195


# ISO data format
import datetime
txt = "2025-04-10T13:43:15+09:00"
dt = datetime.datetime.fromisoformat(txt)
unixtime = dt.timestamp()
print(f"dt: {dt}")
print(f"unixtime: {unixtime}")

tsecs = 1744260195
# with local timezone
dt = datetime.datetime.fromtimestamp(tsecs).astimezone().isoformat()
print(f"dt: {dt}")

dt = datetime.datetime.fromtimestamp(tsecs).astimezone(datetime.timezone.utc).isoformat()
print(f"dt: {dt}")
