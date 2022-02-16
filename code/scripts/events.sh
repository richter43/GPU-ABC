#!/bin/bash

BIN_PATH="../bin"
LOG_PATH="./events"

sudo -E env "PATH=$PATH" nvprof --events all \
        --log-file "${LOG_PATH}/events-$2.log" \
        "${BIN_PATH}/$1 $3 $4 $5"

cat "${LOG_PATH}/events-$2.log" | egrep -i "(warps_launched|not_predicated|sm_cta)"
