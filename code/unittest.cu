#include <stdio.h>
#include <stdlib.h>

#include "benchfuns.h"

#define FLOAT_VAL 1.1 

void h_rastrigin1d_test(float x);
void h_rastriginnd_test(void);

__global__ void rastrigin(float *ret, float x);
__global__ void d_rastriginnd_test(float *ret, float *x, int n);

int main(void){
	
	//h_rastrigin1d_test(FLOAT_VAL);
	h_rastriginnd_test();
	return EXIT_SUCCESS;
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

//Device kernels
__global__ void rastrigin(float *ret, float x){
	*ret = rastrigin_1d(x);
	return;
}

__global__ void d_rastriginnd_test(float *ret, float *x, int n){
	rastrigin_nd(ret, x, n);
}
