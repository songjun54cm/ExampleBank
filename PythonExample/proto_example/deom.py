__author__ = "JunSong<songjun@kuaishou.com>"
# Date: 2020/11/04
import json
import proto_example.merchant_roas_pb2 as merchant_pb
from google.protobuf.json_format import MessageToJson, Parse


redis_conf = merchant_pb.RedisConfig()
redis_conf.redis_pipeline_cluster_name = "adMerchantROASBid"
redis_conf.redis_session_timeout_millis = 20
redis_conf.redis_read_timeout_millis = 20
redis_conf.redis_write_timeout_millis = 20
redis_conf.redis_batch_read_timeout_millis = 40
redis_conf.redis_batch_write_timeout_millis = 40

kafka_conf = merchant_pb.KafkaConfig()
kafka_conf.kafka_topic = "ad_merchant_roas_bid"
kafka_conf.user_params = ""

redis_batch_sender = merchant_pb.BatchSenderConfig()
redis_batch_sender.queue_size = 200000
redis_batch_sender.thread_num = 4
redis_batch_sender.redis_config.CopyFrom(redis_conf)

kafka_batch_sender = merchant_pb.BatchSenderConfig()
kafka_batch_sender.queue_size = 200000
kafka_batch_sender.thread_num = 4
kafka_batch_sender.kafka_config.CopyFrom(kafka_conf)

roas_conf = merchant_pb.MerchantROASBidConfig()
roas_conf.enable_merchant_roas_bid = False
roas_conf.context_sync_time_interval_ms = 60000
roas_conf.result_sync_time_interval_ms = 60000
roas_conf.budget_consume_rate_low_threshold = 0.2
roas_conf.day_charge_amount_low_threshold = 100000
roas_conf.target_roas_fluctuating_rate = 0.2
roas_conf.redis_batch_sender_config.CopyFrom(redis_batch_sender)
roas_conf.kafka_batch_sender_config.CopyFrom(kafka_batch_sender)

roas_conf_json = MessageToJson(roas_conf, preserving_proto_field_name=True)
with open("output.txt", "w") as out_f:
    # out_f.write(MessageToJson(roas_conf, preserving_proto_field_name=True))
    out_f.write(roas_conf_json)

roas_conf_2 = merchant_pb.MerchantROASBidConfig()
Parse(roas_conf_json, roas_conf_2)
print(roas_conf_2)
print(roas_conf_2.SerializeToString())
print(MessageToJson(roas_conf_2))

