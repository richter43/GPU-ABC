#!/bin/bash

LOG_FOLDER="./gpu-trace"
BIN_PATH="../bin"

if [ $# -ne 2 ]
then
        echo "Missing program or log name"
        exit -1
fi

sudo -E env "PATH=$PATH" nvprof --print-gpu-trace "$BIN_PATH/$1" 2> "${LOG_FOLDER}/gpu-trace-$2.log"
