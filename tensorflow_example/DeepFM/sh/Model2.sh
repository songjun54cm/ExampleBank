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
mode=${2} # train/test/pred/train2
hPath="hdfs://hz-cluster4/user/ndir/music_recommend/dailyRecMusic/sample/dfm/F0/2018-10-19"
train_path=${hPath}/train
dev_path=${hPath}/dev
test_path=${hPath}/test
echo '-------------------------------------'

# 3. python
cd ${DIR}/src/

if [ ${mode} == 'pred' ];then
for((index=0;index<4;index++))
    do
      model=dfm2_v${version}_${3}
      gpuid=`expr ${index} + 4 `
      output=${output}"/"${index}".txt"

      echo "version:"${version}
      echo "model :"${model}
      echo "mode  :"${mode}
      echo "gpuid :"${gpuid}
      echo "index :"${index}
      echo "input :"${input}
      echo "output:"${output}

      python Model2.py \
      --version ${version} \
      --model ${model} \
      --mode ${mode} \
      --gpuid ${gpuid} \
      --index ${index} \
      --input ${input} \
      --output ${output}"/"${index}".txt" \
      > /home/ndir/zhangying5/DeepFM/log/Pred${version}-${index}.${today}.log 2>&1 &
    done
elif [ ${mode} == 'train' ];then
    model=dfm2_v${version}_${today}
    gpuid=${3}

    echo "version:"${version}
    echo "model:"${model}
    echo "mode :"${mode}
    echo "gpuid:"${gpuid}
    echo "train:"${train_path}
    echo "dev  :"${dev_path}

    python Model2.py \
    --version ${version} \
    --model ${model} \
    --mode ${mode} \
    --gpuid ${gpuid} \
    --train ${train_path} \
    --dev ${dev_path} \
    > /home/ndir/zhangying5/DeepFM/log/Train${version}.${today}.log 2>&1 &
elif [ ${mode} == 'train2' ];then
    model=dfm2_v${version}_${3}
    gpuid=${4}
    repeat=${5}

    echo "version:"${version}
    echo "model:"${model}
    echo "mode :"${mode}
    echo "gpuid:"${gpuid}
    echo "repeat:"${repeat}
    echo "train:"${train_path}
    echo "dev  :"${dev_path}

    python Model2.py \
    --version ${version} \
    --model ${model} \
    --mode ${mode} \
    --gpuid ${gpuid} \
    --repeat ${repeat} \
    --train ${train_path} \
    --dev ${dev_path} \
    > /home/ndir/zhangying5/DeepFM/log/Train2_${version}.${today}.log 2>&1 &
elif [ ${mode} == 'test' ];then
    model=dfm2_v${version}_${3}
    gpuid=${4}
    if [ $# -gt 4 ];then
      test_path=${hPath}"/"${5}
    fi

    echo "version:"${version}
    echo "model:"${model}
    echo "mode :"${mode}
    echo "gpuid:"${gpuid}
    echo "test :"${test_path}

    python Model2.py \
    --version ${version} \
    --model ${model} \
    --mode ${mode} \
    --gpuid ${gpuid} \
    --test ${test_path} \
    > /home/ndir/zhangying5/DeepFM/log/Test_${model}.log 2>&1 &
fi



if [ $? -ne 0 ];then
    echo "job failed!"
    echo "Model2.py failed" |mail -s "Failed: Model2" $REC_MAILS
    exit 1
fi
echo '-------------------------------------'


STOP_TIME=`date "+%Y-%m-%d %H:%M:%S"`
echo 'Start: '$START_TIME
echo 'Stop : '$STOP_TIME
echo '-------------------------------------'
