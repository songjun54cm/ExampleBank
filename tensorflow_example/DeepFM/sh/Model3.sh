#!/bin/bash
shopt -s expand_aliases
source ${HADOOP_HOME}/libexec/hadoop-config.sh
source /home/ndir/.bash_profile

DIR=/home/ndir/zhangying5/DeepFM
REC_MAILS="zhangying5@corp.netease.com"
START_TIME=`date "+%Y-%m-%d %H:%M:%S"`

# 1. parameters
version=${1}
today=`date +%Y-%m-%d`
mode=${2} # train/test/test2/train2
hPath="hdfs://hz-cluster4/user/ndir/music_recommend/dailyRecMusic/sample/dfm/F1/2018-11-14"
train_path=${hPath}/train
dev_path=${hPath}/dev
test_path=${hPath}/dev
dfm=dfm3
script=Model3.py
echo '-------------------------------------'

# 3. python
cd ${DIR}/src/

if [ ${mode} == 'train' ];then
    model=${dfm}_v${version}_${today}
    gpuid=${3}

    echo "version:"${version}
    echo "model:"${model}
    echo "mode :"${mode}
    echo "gpuid:"${gpuid}
    echo "train:"${train_path}
    echo "dev  :"${dev_path}

    python ${script} \
    --version ${version} \
    --model ${model} \
    --mode ${mode} \
    --gpuid ${gpuid} \
    --train ${train_path} \
    --dev ${dev_path} \
    > /home/ndir/zhangying5/DeepFM/log/Train_${model}.log 2>&1 &
elif [ ${mode} == 'train2' ];then
    model=${dfm}_v${version}_${3}
    gpuid=${4}
    repeat=${5}

    echo "version:"${version}
    echo "model:"${model}
    echo "mode :"${mode}
    echo "gpuid:"${gpuid}
    echo "repeat:"${repeat}
    echo "train:"${train_path}
    echo "dev  :"${dev_path}

    python ${script} \
    --version ${version} \
    --model ${model} \
    --mode ${mode} \
    --gpuid ${gpuid} \
    --repeat ${repeat} \
    --train ${train_path} \
    --dev ${dev_path} \
    > /home/ndir/zhangying5/DeepFM/log/Train2_${model}.log 2>&1 &
elif [ ${mode} == 'test' ];then
    model=${dfm}_v${version}_${3}
    gpuid=${4}
    if [ $# -gt 4 ];then
      test_path=${hPath}"/"${5}
    fi

    echo "version:"${version}
    echo "model:"${model}
    echo "mode :"${mode}
    echo "gpuid:"${gpuid}
    echo "test :"${test_path}

    python ${script} \
    --version ${version} \
    --model ${model} \
    --mode ${mode} \
    --gpuid ${gpuid} \
    --test ${test_path} \
    > /home/ndir/zhangying5/DeepFM/log/Test_${model}.log 2>&1 &
elif [ ${mode} == 'test2' ];then
    model=${dfm}_v${version}_${3}
    gpuid=${4}
    if [ $# -gt 4 ];then
      test_path="hdfs://hz-cluster4/user/ndir/music_recommend/dailyRecMusic/sample/dfm/F0/"${5}/"test"
    fi

    echo "version:"${version}
    echo "model:"${model}
    echo "mode :"${mode}
    echo "gpuid:"${gpuid}
    echo "test :"${test_path}

    python ${script} \
    --version ${version} \
    --model ${model} \
    --mode ${mode} \
    --gpuid ${gpuid} \
    --test ${test_path} \
    > /home/ndir/zhangying5/DeepFM/log/Test2_${model}.log 2>&1 &
fi


if [[ $? -ne 0 ]];then
    echo "job failed!"
    echo "${script} failed" |mail -s "Failed: Model2" $REC_MAILS
    exit 1
fi
echo '-------------------------------------'


STOP_TIME=`date "+%Y-%m-%d %H:%M:%S"`
echo 'Start: '$START_TIME
echo 'Stop : '$STOP_TIME
echo '-------------------------------------'
