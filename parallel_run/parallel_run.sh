#!/bin/bash

PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )/../"
cd $PROJECT_ROOT

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

mkdir -p $LOGS_DIR$MAIN_DIR_NAME/
CONFIGS=$(ls $CONFIGS_DIR)
echo "main directory ---> $MAIN_DIR_NAME"

# Starting parallel
echo starting new processes
rm .jobs_id
for CONF in $CONFIGS
do
  cp $CONFIGS_DIR$CONF app/configs.json
  for (( i=1; i<=$NUMBER_OF_ITERATION ; i++ ))
  do
    echo $CONFIGS_DIR$CONF
    LOG_FILE=${LOGS_DIR}${MAIN_DIR_NAME}/${CONF}${i}.log
    python main.py $MAIN_DIR_NAME &>> $LOG_FILE &
    jobs -p >> .jobs_id
  done
  sleep 5
done

# Keep the processes ID
echo $(cat .jobs_id | wc -l) process created
