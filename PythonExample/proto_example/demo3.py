__author__ = "JunSong<songjun@kuaishou.com>"

# Date: 2020/11/12
import proto_example.merchant_roas_pb2 as merchant_pb
from google.protobuf.json_format import MessageToJson, Parse

flow_map = merchant_pb.FlowControlMap()


user1 = merchant_pb.FlowControl()
user1.user_id = 111222333
user1.limit_ratio = 10
user1.valid_start_time = 111000000
user1.valid_end_time = 222000000

user2 = merchant_pb.FlowControl()
user2.user_id = 222333444
user2.limit_ratio = 20
user2.valid_start_time = 222000000
user2.valid_end_time = 333000000

flow_map.user_info[111222333].CopyFrom(user1)
flow_map.user_info[222333444].CopyFrom(user2)

print(MessageToJson(flow_map, preserving_proto_field_name=True))
