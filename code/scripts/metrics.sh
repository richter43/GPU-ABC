#!/bin/bash

BIN_PATH="../bin"
LOG_PATH="./metrics"

if [ $# -ne 2 ]
then
        echo "Missing program or log name"
        exit -1
fi

LOG_NAME="${LOG_PATH}/metrics-$2.log"

sudo -E env "PATH=$PATH" nvprof --metrics all \
       	--log-file ${LOG_NAME} \
	"${BIN_PATH}/$1"

sed -n 6,8p ${LOG_NAME} > "${LOG_PATH}/solution.log"

cat ${LOG_NAME} | egrep -i "(  ipc|inst_integer|flop_count_sp |inst_control|inst_compute_ld_st|inst_per_warp|shared_efficiency|shared_utilization)"  >> \
	"${LOG_PATH}/solution.log"

cat "${LOG_PATH}/solution.log"
