#ifndef _H_ABC
#define _H_ABC

//Device kernels
extern __device__ void scout_bee(curandState *state, float *sol, int dim, float min, float max);
extern __device__ void initialize_sol_array(curandState *state, float *sol, int sol_dim, float min, float max);
extern __device__ void print_float_sol(float *array, int size);

//Global kernels
extern __global__ void abc_algo(curandState *state, float *sol_array, int array_size, int sol_dim, float min, float max);

#endif
