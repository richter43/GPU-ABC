#!/bin/bash

sudo -E env "PATH=$PATH" nvprof --print-api-trace ./../bin/gpu $1
