#include <stdio.h>
#include <stdlib.h>
#include <helper_functions.h>
#include <helper_cuda.h>

#include <cuda.h>
#include <curand_kernel.h>
#include "utils.h"

//Host utility functions
void *malloc_errorhandler(size_t size){
	void *tmp_malloc = malloc(size);
	if(tmp_malloc == NULL){
		fprintf(stderr, "Error, could not allocate memory in host device\n");
		exit(EXIT_FAILURE);
	}
	return tmp_malloc;
}

curandState *create_random_state(int blocks, int threads, int seed){

	curandState *state;
	checkCudaErrors(cudaMalloc(&state, sizeof(curandState)*blocks*threads));
        setup_kernel<<<blocks, threads>>>(state, seed);

	return state;
}

float *create_sol_array(int rows, int cols){
	float *d_array;
	checkCudaErrors(cudaMalloc(&d_array, sizeof(float)*rows*cols));
	return d_array;
}

//Device utility functions
__global__ void setup_kernel(curandState *state, int seed){
	int id = threadIdx.x + blockIdx.x * blockDim.x;
	//id could be used as seed to generate a different pseudorandom sequence
	curand_init(seed, id, 0, &state[id]);
	return;
}

__device__ float get_random_float(curandState *state, float min, float max){
	float val = curand_uniform(state);
	val = min + val*(max-min);
	return val;
}

__device__ void random_float_array(curandState *state, float *ret, int size, float min, float max){
	for(int i = 0; i < size; i++){
		ret[i] = get_random_float(state, min, max);
	}
	return;
}
