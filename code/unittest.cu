#include <stdio.h>
#include <stdlib.h>
#include <helper_functions.h>
#include <helper_cuda.h>
#include <curand_kernel.h>

#include "benchfuns.h"
#include "utils.h"
#include "abc.h"

#define FLOAT_VAL 1.1 
#define THREADS 32
#define BLOCKS 1
#define MIN_FLOAT -3.0
#define MAX_FLOAT 3.0
#define SEED 0
//Sum array constant
#define MAX_ARRAY 8

void h_rastrigin1d_test(float x);
void h_rastriginnd_test(void);
void h_curand_test(void);
void h_random_float_array(int dim);
void h_sum_array_float(void);

__global__ void rastrigin(float *ret, float x);
__global__ void d_rastriginnd_test(float *ret, float *x, int n);
__global__ void d_get_random_float(curandState *state);
__global__ void d_random_float_array(curandState *state, float *ret, int dim);
__global__ void d_sum_array_float(float *float_array, int dim);


int main(void){
	
	//h_rastrigin1d_test(FLOAT_VAL);
	//h_rastriginnd_test();
	//h_curand_test();
	//h_random_float_array(6);
	h_sum_array_float();
	return EXIT_SUCCESS;
}

void h_curand_test(void){
	
	curandState *state;

	checkCudaErrors(cudaMalloc(&state, sizeof(curandState)));
	
	setup_kernel<<<BLOCKS, THREADS>>>(state, SEED);
	d_get_random_float<<<BLOCKS, THREADS>>>(state);

	cudaFree(state);
	return;
}

//Host functions
void h_rastrigin1d_test(float x){
	//Host code
	float dev_res;
	//Device code
	float *d_ret;
	cudaMalloc(&d_ret, sizeof(float));
	//Executing kernel
	rastrigin<<<1,1>>>(d_ret, x);

	cudaMemcpy(&dev_res, d_ret, sizeof(float), cudaMemcpyDeviceToHost);
	cudaFree(d_ret);

	float host_res = rastrigin_1d(x);

	printf("Computed at host: %f\n", host_res);
	printf("Computed at device: %f\n", dev_res);
	printf("Distance between the two %.12f\n", host_res - dev_res);
	
	return;
}

void h_rastriginnd_test(void){

	//Host code
	float h_res, initvals[2];
	initvals[0] = 0.0f;
	initvals[1] = 0.0f;

	//Device code
	float *d_ret, *d_initvals;
	cudaMalloc(&d_ret, sizeof(float));
	cudaMalloc(&d_initvals, 2*sizeof(float));

	cudaMemcpy(d_initvals, initvals, 2*sizeof(float), cudaMemcpyHostToDevice);	

	//Executing kernel
	d_rastriginnd_test<<<1,1>>>(d_ret, d_initvals, 2);

	cudaMemcpy(&h_res, d_ret, sizeof(float), cudaMemcpyDeviceToHost);

	cudaFree(d_ret);
	cudaFree(d_initvals);

	float host_res;
	rastrigin_nd(&host_res,initvals,2);
	
	printf("Computed at host: %f\n", host_res);
	printf("Computed at device: %f\n", h_res);
	printf("Distance between the two %.12f\n", host_res - h_res);
	
	return;
}

void h_random_float_array(int dim){
	//Tests the device's ability of creating an array of floats
	//Creates a random state
	curandState *state = create_random_state(BLOCKS, THREADS, SEED);
	//Memory allocation of host variables
	float *h_ret = (float*) malloc(sizeof(float)*dim);
	//Memory allocation of device variables 
	float *d_ret; 
	checkCudaErrors(cudaMalloc(&d_ret, sizeof(float)*dim));
	//Device kernel execution
	d_random_float_array<<<1,1>>>(state, d_ret, dim);
	//Copying memory to host
	checkCudaErrors(cudaMemcpy(h_ret, d_ret, sizeof(float)*dim, cudaMemcpyDeviceToHost));
	//Printing final result
	for(int i = 0; i < dim; i++){
		printf("%f ", h_ret[i]);
	}
	printf("\n");
	//Freeing dynamically allocated memory
	free(h_ret);
	cudaFree(state);
	cudaFree(d_ret);
	return;
}

void h_sum_array_float(void){

	float float_array[MAX_ARRAY];
	float counter = 0.0;

	for(int i = 0; i < MAX_ARRAY; i++){
		float_array[i] = counter;
		counter += 1.0;
	}

	float *d_float_array;
	checkCudaErrors(cudaMalloc(&d_float_array, sizeof(float)*MAX_ARRAY));
	checkCudaErrors(cudaMemcpy(d_float_array, float_array, sizeof(float)*MAX_ARRAY, cudaMemcpyHostToDevice));

	d_sum_array_float<<<1, MAX_ARRAY>>>(d_float_array, MAX_ARRAY);

	cudaFree(d_float_array);
	return;
}

__global__ void d_sum_array_float(float *float_array, int dim){

	int id = threadIdx.x + blockIdx.x * blockDim.x;

	float res = sum_fitness_naive(float_array, dim, id);
	if(id == 0){
		printf("Sum of the array is equal to: %f\n", res);
	}
	
	return;
}
//Device kernels
__global__ void rastrigin(float *ret, float x){
	*ret = rastrigin_1d(x);
	return;
}

__global__ void d_rastriginnd_test(float *ret, float *x, int n){
	rastrigin_nd(ret, x, n);
}

__global__ void d_get_random_float(curandState *state){
	float val = get_random_float(state, MIN_FLOAT, MAX_FLOAT);
        int id = threadIdx.x + blockIdx.x * blockDim.x;
        printf("%d %f\n", id, val);
	return;
}

__global__ void d_random_float_array(curandState *state, float *ret, int dim){
	random_float_array(state, ret, dim, MIN_FLOAT, MAX_FLOAT);
	return;
}
