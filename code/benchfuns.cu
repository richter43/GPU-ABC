#include "benchfuns.h"

#define AMPLITUDE_RASTRIGIN 10.0

__host__ __device__ void sphere_nd(float *ret, float *x, int n){
	float sum = 0;
	for(int i = 0; i < n; i++){
		sum += powf(x[i],2.0);
	}
	*ret = sum;
	return;
}

__host__ __device__ float rastrigin_1d(float x){
	return AMPLITUDE_RASTRIGIN + (powf(x, 2.0) - AMPLITUDE_RASTRIGIN*cospif(2.0*x));
}

__host__ __device__ void rastrigin_nd(float *ret, float *x, int n){
	float sum = AMPLITUDE_RASTRIGIN * n;
	for(int i = 0; i < n; i++){
		sum += powf(x[i],2.0) - AMPLITUDE_RASTRIGIN*cospif(2.0*x[i]);	
	}
	*ret = sum;
	return;
}
