#!/bin/bash

if [ $# -ne 2 ]
then
        echo "Missing program or log name"
        exit -1
fi

mkdir -p ./events ./gpu-trace ./events ./metrics &> /dev/null

sh gpu-trace.sh $1 $2
sh api-trace.sh $1 $2
sh events.sh $1 $2
sh metrics.sh $1 $2
