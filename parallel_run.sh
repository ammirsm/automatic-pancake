#!/bin/bash

# Kill other processes that run before
cat .jobs_id | xargs kill &> /dev/null
if [[ $? ==  0 ]]
then
  cat .jobs_id | xargs echo
  cat /dev/null > .jobs_id
  echo all processes has stopped
fi
if [[ $1 == 'stop' ]]
then
  exit 0
fi


# Configs for runnong parallel
set -e

. parallel_config.env

mkdir -p $MAIN_DIR_NAME$LOGS_DIR
CONFIGS=$(ls $CONFIGS_DIR)
echo "main directory ---> $MAIN_DIR_NAME"

# Starting parallel
echo starting new processes
for CONF in $CONFIGS
do
  cp $CONFIGS_DIR$CONF app/configs.py
  for (( i=1; i<=$NUMBER_OF_ITERATION ; i++ ))
  do
#    echo $CONFIGS_DIR$CONF
    LOG_FILE=${MAIN_DIR_NAME}${LOGS_DIR}${CONF}${i}.log
    python main.py $MAIN_DIR_NAME &>> $LOG_FILE &
  done
  sleep 5
done

# Keep the processes ID
jobs -p > .jobs_id
echo $(cat .jobs_id | wc -l) process created
