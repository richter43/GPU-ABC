#include "benchfuns.h"

#define AMPLITUDE 10.0

__host__ __device__ float rastrigin_1d(float x){
	return AMPLITUDE + (powf(x, 2.0) - AMPLITUDE*cospif(2.0*x));
}

__host__ __device__ void rastrigin_nd(float *ret, float *x, int n){
	float sum = AMPLITUDE * n;
	for(int i = 0; i < n; i++){
		sum += powf(x[i],2.0) - AMPLITUDE*cospif(2.0*x[i]);	
	}
	*ret = sum;
	return;
}
