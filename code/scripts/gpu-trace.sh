#!/bin/bash


sudo -E env "PATH=$PATH" nvprof --print-gpu-trace ../bin/gpu $1
