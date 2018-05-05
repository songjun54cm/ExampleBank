"""
Author: songjun
Date: 2018/4/23
Description:
Usage:
"""
import argparse
from kafka import KafkaProducer
from kafka.errors import KafkaError
import logging
from settings import BOOTSTRAP_SERVERS

def on_send_success(record_metadata):
    print('send success: topic: %s, partition: %s, offset: %s' %
          (str(record_metadata.topic), str(record_metadata.partition), str(record_metadata.offset)))
    # print(record_metadata.topic)
    # print(record_metadata.partition)
    # print(record_metadata.offset)


def on_send_error(excp):
    logging.error("An errorback", excp=excp)


def main(config):
    producer = KafkaProducer(bootstrap_servers=BOOTSTRAP_SERVERS,
                             retries=3)
    for i in range(10):
        message = b"message_%d" % i
        producer.send('songjun_topic', message)\
            .add_callback(on_send_success)\
            .add_errback(on_send_error)
        # print('send message : %s' % message)

    producer.flush()
    print('producer send message finish.')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file', dest='file', type=str, default='example.txt')
    args = parser.parse_args()
    config = vars(args)
    main(config)
