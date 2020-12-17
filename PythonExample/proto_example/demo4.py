__author__ = "JunSong<songjun@kuaishou.com>"

# Date: 2020/11/12
import proto_example.merchant_roas_pb2 as merchant_pb
from google.protobuf.json_format import MessageToJson, Parse

flow_map = merchant_pb.FlowControlMapV2()


user1 = merchant_pb.FlowControlInfoV2()
user1.config_id = 111222333
user1.author_id.append(1)
user1.author_id.append(2)
user1.limit_ratio = 10
user1.valid_start_time = 111000000
user1.valid_end_time = 222000000

user2 = merchant_pb.FlowControlInfoV2()
user2.config_id = 222333444
user2.limit_ratio = 20
user2.valid_start_time = 222000000
user2.valid_end_time = 333000000

flow_map.config_map[111222333].CopyFrom(user1)
flow_map.config_map[222333444].CopyFrom(user2)

# print(MessageToJson(flow_map, preserving_proto_field_name=True))

author_list = [
  403082302,
  188888880,
  544447457,
  74405960,
  8322582,
  1118162755,
  140673020,
  151344124,
  10150585,
  704974956,
  8110234,
  529984242,
  1016355266,
  1082820781,
  1040816918,
  265046537,
  1524700320,
  719084479,
  908750763,
  763377077,
  1377177112,
  764461430,
  1045912089,
  54139598,
  1204122107,
  5875841,
  98878703,
  1132841810,
  1169301317,
  492122351,
  374020066,
  1331653953,
  1851331545,
  130419414,
  1474014320,
  6105884,
  780243981,
  2863872,
  138177741,
  267133045,
  31529737,
  53100097,
  1216999122,
  991654789,
  1135131550,
  1085449620,
  656004318,
  1343499387,
  1786976929
]

flow_map = merchant_pb.FlowControlMapV2()

conf1 = merchant_pb.FlowControlInfoV2()
conf1.config_id = 1
for author_id in author_list:
    conf1.author_id.append(author_id)
conf1.limit_ratio = 10
conf1.valid_start_time = 1606752000000
conf1.valid_end_time = 0

flow_map.config_map[conf1.config_id].CopyFrom(conf1)

print(MessageToJson(flow_map, preserving_proto_field_name=True))