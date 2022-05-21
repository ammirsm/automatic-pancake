#!/bin/bash

cat .jobs_id | xargs kill &> /dev/null
if [[ $? ==  0 ]]
then
  cat .jobs_id | xargs echo
  echo /dev/null > .jobs_id
  echo all processes has stopped
fi
if [[ $1 == 'stop' ]]
then
  exit 0
fi

set -e

NUMBER_OF_ITERATION=5

LOGS_DIR=./logs/
mkdir -p $LOGS_DIR



echo starting new processes
for (( i=1; i<=$NUMBER_OF_ITERATION ; i++ ))
do
  LOG_FILE=${LOGS_DIR}${i}.log
  python main.py >> $LOG_FILE &
#  sleep 20 &
done

jobs -p > .jobs_id
