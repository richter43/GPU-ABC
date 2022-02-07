#Compiler and necessary libraries
CUDA_PATH := /usr/local/cuda
NVCC := ${CUDA_PATH}/bin/nvcc
INCLUDES := -I ${CUDA_PATH}/samples/common/inc

#Files and objects 
DEST_FOLDER := ./bin
ANCILLARY_OBJS := benchfuns.o utils.o abc.o
TEST_OBJS := ${ANCILLARY_OBJS} unittest.o
MAIN_OBJS := ${ANCILLARY_OBJS} main.o

all: gpu

gpu: ${MAIN_OBJS} | folder
	${NVCC} ${INCLUDES} $^ -o ${DEST_FOLDER}/$@ -g

folder:
	mkdir -p ${DEST_FOLDER}

unit_test: ${TEST_OBJS} | folder
	${NVCC} ${INCLUDES} $^ -o ${DEST_FOLDER}/$@ -g

clean:
	rm -rf ${DEST_FOLDER}

%.o: %.cu
	${NVCC} ${INCLUDES} -dc $< -o $@
