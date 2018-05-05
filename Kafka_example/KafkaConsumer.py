"""
Author: songjun
Date: 2018/4/23
Description:
Usage:
"""
import argparse
from kafka import KafkaConsumer, TopicPartition
from settings import BOOTSTRAP_SERVERS

def main_test1():
    consumer = KafkaConsumer('songjun_topic',
                             group_id='songjun_group',
                             bootstrap_servers=BOOTSTRAP_SERVERS,
                             consumer_timeout_ms=100000,
                             )
    print('start getting message')
    for message in consumer:
        print("%s:%d:%d: key=%s value=%s" % (message.topic, message.partition,
                                             message.offset, message.key,
                                             message.value))

def main_test2():
    consumer = KafkaConsumer('songjun_topic',
                             group_id='songjun_group_11',
                             bootstrap_servers=BOOTSTRAP_SERVERS,
                             # consumer_timeout_ms=10000,
                             auto_offset_reset='earliest')
    print('start getting message')
    for message in consumer:
        print("%s:%d:%d: key=%s value=%s" % (message.topic, message.partition,
                                             message.offset, message.key,
                                             message.value))

def main(config):
    main_test2()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file', dest='file', type=str, default='example.txt')
    args = parser.parse_args()
    config = vars(args)
    main(config)

