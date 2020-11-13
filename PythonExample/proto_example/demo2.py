__author__ = "JunSong<songjun@kuaishou.com>"

# Date: 2020/11/5
import proto_example.merchant_roas_pb2 as merchant_pb
from google.protobuf.json_format import MessageToJson, Parse


punish_conf = merchant_pb.TopAuthorPunishConfig()
thr_conf1 = punish_conf.cpm_thr_conf.add()
thr_conf1.author_id = 123
thr_conf1.cpm_thr_pos = 20
thr_conf2 = punish_conf.cpm_thr_conf.add()
thr_conf2.author_id = 123456
thr_conf2.cpm_thr_pos = 10
punish_conf.cpm_thr_conf_map[123].CopyFrom(thr_conf1)
punish_conf.cpm_thr_conf_map[123456].CopyFrom(thr_conf2)
print()