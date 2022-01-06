#include <stdio.h>
#include <stdlib.h>

#include "benchfuns.h"

#define DEBUG 1

#define FLOAT_VAL 1.1 

void test_rastrigin1d(float x);

__global__ void rastrigin(float *ret, float x);

int main(void){
	
	test_rastrigin1d(FLOAT_VAL);

	return EXIT_SUCCESS;
}

//Host functions
void test_rastrigin1d(float x){
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
	
	//
	printf("Computed at host: %f\n", host_res);
	printf("Computed at device: %f\n", dev_res);
	printf("Distance between the two %.12f\n", host_res - dev_res);
	
	return;
}

//Device kernels
__global__ void rastrigin(float *ret, float x){
	*ret = rastrigin_1d(x);
	return;
}
