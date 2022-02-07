#ifndef _H_UTILS
#define _H_UTILS

#include <curand_kernel.h>

//Host functions
extern void *malloc_errorhandler(size_t size);
extern float *create_sol_array(int rows, int cols);
extern curandState *create_random_state(int blocks, int threads, int seed); 

//Device kernels
extern __global__ void setup_kernel(curandState *state, int seed);
extern __device__ float get_random_float(curandState *state, float min, float max);
extern __device__ void random_float_array(curandState *state, float *ret, int size, float min, float max);

#endif
