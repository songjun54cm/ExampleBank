"""
Author: songjun
Date: 2018/4/25
Description:
Usage:
"""
import nos
import json
from NOS_PY.settings import NOS_ACCESS_KEY, NOS_ENDPOINT, NOS_ACCESS_SECRET, NOS_BUCKET, NOS_OBJECT_KEY_PREFIX

client = nos.Client(
    access_key_id=NOS_ACCESS_KEY,
    access_key_secret=NOS_ACCESS_SECRET,
    end_point=NOS_ENDPOINT,
)
# song_id = "songjun_test_song"
song_id = "472804924"
bucket = NOS_BUCKET
object_key = NOS_OBJECT_KEY_PREFIX + song_id
try:
    resp = client.get_object(
        bucket=bucket,
        key=object_key,
    )
    print(resp)
    body = resp['body']
    feature_info = json.loads(body.read())
    print("Bucket: %s, Key: %s" % (bucket, object_key))
    for key in feature_info.keys():
        val = feature_info[key]
        if isinstance(val, list):
            first_val = val[0]
            last_val = val[-1]
            size_str = str(len(val))
            if isinstance(first_val, list):
                size_str += " * " + str(len(first_val))
                first_val = first_val[0]
                last_val = last_val[-1]
            val_str = "size: <%s>, first: %s, last: %s" % (size_str, str(first_val), str(last_val))
        else:
            val_str = str(val)

        print('key: %s, type: %s, value: %s' % (key, str(type(feature_info[key])), val_str))

except nos.exceptions.ServiceException as e:
   print (
       'ServiceException: %s\n'
       'status_code: %s\n'
       'error_type: %s\n'
       'error_code: %s\n'
       'request_id: %s\n'
       'message: %s\n'
   ) % (
       e,
       e.status_code,  # error http code
       e.error_type,   # NOS server error type
       e.error_code,   # NOS server error code
       e.request_id,   # request id
       e.message       # error description message
   )
except nos.exceptions.ClientException as e:
   print (
       'ClientException: %s\n'
       'message: %s\n'
   ) % (
       e,
       e.message       # client error message
   )

print('get NOS fihish.')

