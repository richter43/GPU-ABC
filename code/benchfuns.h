#ifndef _H_BENCHFUN
#define _H_BENCHFUN

extern __host__ __device__ float rastrigin_1d(float x);
extern __host__ __device__ void rastrigin_nd(float *ret, float *x, int n);

#endif
