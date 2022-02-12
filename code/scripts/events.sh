#!/bin/bash

BIN_PATH="/home/seivarden/Labs/Lab02/Ex01/bin"
LOG_PATH="/home/seivarden/Labs/Lab02/Ex01/logs"

if [ $# -ne 1 ]
then
        echo "Missing program name"
        exit -1
fi

sudo -E env "PATH=$PATH" nvprof --events all \
        --log-file "${LOG_PATH}/events.log" \
        "${BIN_PATH}/$1"

cat "${LOG_PATH}/events.log" | egrep -i "(warps_launched|not_predicated|sm_cta)"
