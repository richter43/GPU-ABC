#include "benchfuns.h"

#define AMPLITUDE_RASTRIGIN 10.0
#define ROSENBROCK_MULTIPLIER 100.0

__host__ __device__ void rosenbrock_nd(float *ret, float *x, int n){
	//Rosenbrock function of an n-dimensional solution

	float sum = 0.0;
	for(int i = 0; i < n-1; i++){
		sum += powf(1.0 - x[i], 2.0) + ROSENBROCK_MULTIPLIER * powf(x[i+1] - powf(x[i], 2.0), 2.0);
	}
	*ret = sum;
	return;
}

__host__ __device__ void sphere_nd(float *ret, float *x, int n){
	//Sphere function of an n-dimensional solution
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
	//Rastrigin function of an n-dimensional solution
	float sum = AMPLITUDE_RASTRIGIN * n;
	for(int i = 0; i < n; i++){
		sum += powf(x[i],2.0) - AMPLITUDE_RASTRIGIN*cospif(2.0*x[i]);	
	}
	*ret = sum;
	return;
}
