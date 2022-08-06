#!/bin/bash

SCRIPT_PATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="${SCRIPT_PATH}/../"

cd $PROJECT_ROOT

# Kill other processes that run before
cat .jobs_id 2&> /dev/null | xargs kill &> /dev/null
if [[ $? ==  0 ]]
then
  cat .jobs_id | xargs echo
  cat /dev/null > $SCRIPT_PATH/.jobs_id
  echo all processes has stopped
fi
if [[ $1 == 'stop' ]]
then
  exit 0
fi


# Configs for runnong parallel
set -e

. $SCRIPT_PATH/parallel_config.env

mkdir -p $SCRIPT_PATH/$LOGS_DIR$MAIN_DIR_NAME/
CONFIGS=$(ls $SCRIPT_PATH/$CONFIGS_DIR)
echo "main directory ---> $MAIN_DIR_NAME"

# Starting parallel
rm $SCRIPT_PATH/.jobs_id 2&> /dev/null || true
for CONF in $CONFIGS
do
  cp $SCRIPT_PATH/$CONFIGS_DIR$CONF app/configs.json
  for (( i=1; i<=$NUMBER_OF_ITERATION ; i++ ))
  do
    echo starting $SCRIPT_PATH/$CONFIGS_DIR$CONF $i
    LOG_FILE=$SCRIPT_PATH/${LOGS_DIR}${MAIN_DIR_NAME}/${CONF}${i}.log
    python main.py $MAIN_DIR_NAME &>> $LOG_FILE &
    jobs -p >> $SCRIPT_PATH/.jobs_id
  done
  sleep 5
done

# Keep the processes ID
echo $(cat $SCRIPT_PATH/.jobs_id | wc -l) process created
