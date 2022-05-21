#!/bin/bash

set -e

NUMBER_OF_ITERATION=5

LOGS_DIR=./logs/
mkdir -p $LOGS_DIR

for (( i=1; i<=$NUMBER_OF_ITERATION ; i++ ))
do
  LOG_FILE=${LOGS_DIR}${i}.log
  python main.py >> $LOG_FILE &
done
