#Compiler and necessary libraries
CUDA_PATH := /usr/local/cuda
NVCC := ${CUDA_PATH}/bin/nvcc
INCLUDES := -I ${CUDA_PATH}/samples/common/inc

#Files and objects 
DEST_FOLDER := ./bin
ANCILLARY_OBJS := benchfuns.o utils.o abc.o
TEST_OBJS := ${ANCILLARY_OBJS} unittest.o
MAIN_OBJS := ${ANCILLARY_OBJS} main.o
DELETE_FILES := *.cudafe1.* *.fatbin.* *.reg.* *.module_id *.ii *.ptx *.cubin *.o *.fatbin

ifeq (${DEBUG}, true)
	DEBUG_OPTIONS := -g -G -keep
else
	DEBUG_OPTIONS :=
endif

all: gpu

gpu: ${MAIN_OBJS} | folder
	${NVCC} ${DEBUG_OPTIONS} ${INCLUDES} $^ -o ${DEST_FOLDER}/$@

folder:
	mkdir -p ${DEST_FOLDER}

unit_test: ${TEST_OBJS} | folder
	${NVCC} ${INCLUDES} $^ -o ${DEST_FOLDER}/$@ -g

clean:
	rm -rf ${DEST_FOLDER}
	rm ${DELETE_FILES}

%.o: %.cu
	${NVCC} ${DEBUG_OPTIONS} ${INCLUDES} -dc $< -o $@
