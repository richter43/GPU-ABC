#!/bin/bash

BIN_PATH="../bin"
LOG_PATH="./events"

if [ $# -ne 2 ]
then
        echo "Missing program or log name"
        exit -1
fi

sudo -E env "PATH=$PATH" nvprof --events all \
        --log-file "${LOG_PATH}/events-$2.log" \
        "${BIN_PATH}/$1"

cat "${LOG_PATH}/events-$2.log" | egrep -i "(warps_launched|not_predicated|sm_cta)"
