"""
Author: songjun
Date: 2018/4/25
Description:
Usage:
"""
import nos
import json
from NOS_PY.settings import NOS_ACCESS_KEY, NOS_ENDPOINT, NOS_ACCESS_SECRET, NOS_BUCKET, NOS_OBJECT_KEY_PREFIX


def main_2():
    import struct
    client = nos.Client(
        access_key_id=NOS_ACCESS_KEY,
        access_key_secret=NOS_ACCESS_SECRET,
        end_point=NOS_ENDPOINT,
    )
    song_id = "songjun_test_song_1"
    bucket = NOS_BUCKET
    object_key = "cnn_feature_" + song_id
    resp = client.get_object(
        bucket=bucket,
        key=object_key,
    )
    body = resp['body']
    spos = 0
    epos = 4
    ndim = struct.unpack('i', body[spos:epos])
    spos = epos
    epos = spos + 4*ndim
    fea_size = struct.unpack('i', body[spos:epos])
    fea_num = 1
    for v in fea_size:
        fea_num *= v
    spos = epos
    epos = spos + fea_num
    feature = struct.unpack('f'*fea_num, body[spos:epos])
    print(feature[0:10])

def main_1():
    client = nos.Client(
        access_key_id=NOS_ACCESS_KEY,
        access_key_secret=NOS_ACCESS_SECRET,
        end_point=NOS_ENDPOINT,
    )
    # song_id = "songjun_test_song"
    song_id = "1002166"
    key_prefix = "meta_"
    bucket = NOS_BUCKET
    object_key = key_prefix + song_id
    try:
        resp = client.get_object(
            bucket=bucket,
            key=object_key,
        )
        print(type(resp))
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



if __name__ == '__main__':
    main_1()



"""
main_1:
song_id = "986433"
<type 'dict'>
{'content_length': 135, 'x_nos_request_id': '6cc3c18a-b4fc-4892-8b29-8b63bcd2c9d8', 'body': <urllib3.response.HTTPResponse object at 0x00000000023FEB38>, 'etag': '84d8db3dc0b36ca0ed014c4951ec9be8', 'content_range': '', 'content_type': 'application/octet-stream;charset=UTF-8'}
Bucket: music-content-feature, Key: feature_986433
key: url, type: <type 'unicode'>, value: ymusic/b7be/9288/d062/34bc560d94cf6bd26890f724edc56879.mp3
key: songId, type: <type 'unicode'>, value: 986433
key: dfsId, type: <type 'NoneType'>, value: None
key: type, type: <type 'unicode'>, value: NOS
key: time, type: <type 'int'>, value: 226000
get NOS fihish.
"""