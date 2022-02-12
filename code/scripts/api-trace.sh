#!/bin/bash

LOG_FOLDER="./api-trace"
BIN_PATH="../bin"

if [ $# -ne 2 ]
then
        echo "Missing program or log name"
        exit -1
fi

sudo -E env "PATH=$PATH" nvprof --print-api-trace "$BIN_PATH/$1" 2> "${LOG_FOLDER}/api-trace-$2.log"
