#!/bin/bash
shopt -s expand_aliases
source ${HADOOP_HOME}/libexec/hadoop-config.sh
source /home/ndir/.bash_profile

DIR=/home/ndir/zhangying5/DeepFM
REC_MAILS="zhangying5@corp.netease.com"
START_TIME=`date "+%Y-%m-%d %H:%M:%S"`

# 1. parameters
model=${1}     # dfm3_v1_2018-11-16
ckpt_v=${2}    # v1-780000
gpuid=${3}
mode=${4}
echo "model:"${model}
echo "ckpt_v:"${ckpt_v}
echo "gpuid :"${gpuid}
echo "mode:"${mode}
echo '-------------------------------------'

# 3. python
cd ${DIR}/src/

python ckpt2pb.py \
--model ${model} \
--ckpt_v ${ckpt_v} \
--gpuid ${gpuid} \
--mode ${mode}


if [ $? -ne 0 ];then
    echo "job failed!"
    echo "ckpt2pb.py failed" |mail -s "Failed: ckpt2pb" $REC_MAILS
    exit 1
fi
echo '-------------------------------------'


STOP_TIME=`date "+%Y-%m-%d %H:%M:%S"`
echo 'Start: '$START_TIME
echo 'Stop : '$STOP_TIME
echo '-------------------------------------'
