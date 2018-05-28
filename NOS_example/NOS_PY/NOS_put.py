"""
Author: songjun
Date: 2018/4/24
Description:
Usage:
"""
import nos
import numpy as np
import json
from NOS_PY.settings import NOS_ACCESS_KEY, NOS_ENDPOINT, NOS_ACCESS_SECRET, NOS_BUCKET, NOS_OBJECT_KEY_PREFIX


def main_2():
    import tempfile
    import struct
    client = nos.Client(
        access_key_id=NOS_ACCESS_KEY,
        access_key_secret=NOS_ACCESS_SECRET,
        end_point=NOS_ENDPOINT,
    )
    song_id = "songjun_test_song_1"
    bucket = NOS_BUCKET
    object_key = "cnn_feature_" + song_id
    feature = np.random.random(256).flatten().tolist()
    print(feature[0:10])
    tempf = tempfile.TemporaryFile('wb')
    tempf.write(struct.pack('i', 1))
    tempf.write(struct.pack('i', 256))
    tempf.write(struct.pack('f'*len(feature), *feature))

    resp = client.put_object(
        bucket=bucket,
        key=object_key,
        body=tempf
    )

    print('put NOS fihish.')
    tempf.close()

def main_1():
    client = nos.Client(
        access_key_id=NOS_ACCESS_KEY,
        access_key_secret=NOS_ACCESS_SECRET,
        end_point=NOS_ENDPOINT,
    )

    song_id = "songjun_test_song"
    feature_info = {
        "audio_id": song_id,
        "version": "python",
        "audio_url": "http://url.com",
        "log_power_mel_feature": np.random.random((10, 3)).tolist(),
        "cnn_feature": np.random.random(20).tolist()
    }

    try:
        resp = client.put_object(
            bucket=NOS_BUCKET,
            key=NOS_OBJECT_KEY_PREFIX + song_id,
            body=json.dumps(feature_info),
        )
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

    print('put NOS fihish.')


if __name__ == '__main__':
    main_2()