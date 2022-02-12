#!/bin/bash

BIN_PATH="."
LOG_PATH="."

if [ $# -ne 1 ]
then
	echo "Missing program name"
	exit -1
fi

sudo -E env "PATH=$PATH" nvprof --metrics all \
       	--log-file "${LOG_PATH}/prof.log" \
	"${BIN_PATH}/$1"

sed -n 6,8p "${LOG_PATH}/prof.log" > "${LOG_PATH}/solution.log"

cat "${LOG_PATH}/prof.log" | egrep -i "(  ipc|inst_integer|flop_count_sp |inst_control|inst_compute_ld_st|inst_per_warp)"  >> \
	"${LOG_PATH}/solution.log"

cat "${LOG_PATH}/solution.log"
