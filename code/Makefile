#Compiler and necessary libraries
CUDA_PATH := /usr/local/cuda
NVCC := ${CUDA_PATH}/bin/nvcc
INCLUDES := -I ${CUDA_PATH}/samples/common/inc

#Files and objects 
DEST_FOLDER := ./bin
ANCILLARY_OBJS := benchfuns.o
TEST_OBJS := ${ANCILLARY_OBJS} unittest.o

all: gpu

object:
	mkdir -p ${DEST_FOLDER}

unit_test: ${TEST_OBJS} | object	
	${NVCC} ${INCLUDES} $^ -o ${DEST_FOLDER}/$@ -g

clean:
	rm -rf ${DEST_FOLDER}

%.o: %.cu
	${NVCC} ${INCLUDES} -dc $< -o $@
