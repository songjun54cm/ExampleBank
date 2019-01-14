#!/bin/bash
shopt -s expand_aliases
source ${HADOOP_HOME}/libexec/hadoop-config.sh
source /home/ndir/.bash_profile

DIR=/home/ndir/zhangying5/DeepFM
REC_MAILS="zhangying5@corp.netease.com"
START_TIME=`date "+%Y-%m-%d %H:%M:%S"`

# 1. parameters
model=${1}
version=${2}
today=`date +%Y-%m-%d`
mode=${3} # train/test/train2/test2
dfm=dfm${model}
script=Model${model}.py
if [[ ${model} -eq 4 ]];then
    hPath="hdfs://hz-cluster4/user/ndir/music_recommend/dailyRecMusic/sample/dfm/F0/2018-10-19"
elif [[ ${model} -eq 4 ]];then
    hPath="hdfs://hz-cluster4/user/ndir/music_recommend/dailyRecMusic/sample/dfm/F1/2018-11-14"
elif [[ ${model} -eq 5 ]];then
    #hPath="hdfs://hz-cluster4/user/ndir/music_recommend/dailyRecMusic/sample/dfm/F2/2018-11-25"
    hPath="hdfs://hz-cluster4/user/ndir/music_recommend/dailyRecMusic/sample/dfm/F4/2018-11-25"
elif [[ ${model} -eq 6 ]];then
    if [[ ${version} -eq 1 ]];then
        #hPath="hdfs://hz-cluster4/user/ndir/music_recommend/dailyRecMusic/sample/dfm/F3/2018-11-25"
        #hPath="hdfs://hz-cluster4/user/ndir/music_recommend/dailyRecMusic/sample/dfm/F3/2018-12-12"
        hPath="hdfs://hz-cluster4/user/ndir/music_recommend/dailyRecMusic/sample/dfm/F3/2018-12-19"
    elif [[ ${version} -eq 3 ]];then
        #hPath="hdfs://hz-cluster4/user/ndir/music_recommend/dailyRecMusic/sample/dfm/F5/2018-12-06"
        hPath="hdfs://hz-cluster4/user/ndir/music_recommend/dailyRecMusic/sample/dfm/F5/2018-12-12"
        #hPath="hdfs://hz-cluster4/user/ndir/music_recommend/dailyRecMusic/sample/dfm/F5/2019-01-01"
    elif [[ ${version} -eq 4 ]];then
        hPath="hdfs://hz-cluster4/user/ndir/music_recommend/dailyRecMusic/sample/dfm/F6/2019-01-07"
    elif [[ ${version} -eq 5 ]];then
        hPath="hdfs://hz-cluster4/user/ndir/music_recommend/dailyRecMusic/sample/dfm/F7/2018-12-24"
    elif [[ ${version} -eq 6 ]];then
        hPath="hdfs://hz-cluster4/user/ndir/music_recommend/dailyRecMusic/sample/dfm/F8/2019-01-01_0"
    elif [[ ${version} -eq 7 ]];then
        hPath="hdfs://hz-cluster4/user/ndir/music_recommend/dailyRecMusic/sample/dfm/F8/2019-01-01_1"
    fi
elif [[ ${model} -eq 7 ]];then
    dfm=dnn${model}
    if [[ ${version} -eq 1 ]];then
        hPath="hdfs://hz-cluster4/user/ndir/music_recommend/dailyRecMusic/sample/dfm/F3/2018-12-12"
    elif [[ ${version} -eq 2 ]];then
        hPath="hdfs://hz-cluster4/user/ndir/music_recommend/dailyRecMusic/sample/dfm/F5/2018-12-12"
    fi
elif [[ ${model} -eq 8 ]];then
    dfm=ncf${model}
    hPath="hdfs://hz-cluster4/user/ndir/music_recommend/dailyRecMusic/sample/dfm/F5/2019-01-01"
elif [[ ${model} -eq 9 ]];then
    dfm=ncf${model}
    hPath="hdfs://hz-cluster4/user/ndir/music_recommend/dailyRecMusic/sample/dfm/F5/2019-01-01"
fi

train_path=${hPath}/train
dev_path=${hPath}/dev
test_path=${hPath}/dev
echo '-------------------------------------'

# 3. python
cd ${DIR}/src/

if [[ ${mode} == 'train' ]];then
    model=${dfm}_v${version}_${today}
    gpuid=${4}

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
elif [[ ${mode} == 'train2' ]];then
    model=${dfm}_v${version}_${4}
    gpuid=${5}
    repeat=${6}

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
elif [[ ${mode} == 'test' ]];then
    model=${dfm}_v${version}_${4}
    gpuid=${5}
    if [[ $# -gt 5 ]];then
      test_path=${hPath}"/"${6}
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
fi



if [[ $? -ne 0 ]];then
    echo "job failed!"
    echo "${script} failed" |mail -s "Failed: ${script}" $REC_MAILS
    exit 1
fi
echo '-------------------------------------'


STOP_TIME=`date "+%Y-%m-%d %H:%M:%S"`
echo 'Start: '$START_TIME
echo 'Stop : '$STOP_TIME
echo '-------------------------------------'
