#ifndef _H_ABC
#define _H_ABC

//Struct
typedef struct abc_info_s{
        //Random state
        curandState *state;
        //Array that contains the solution matrix (Flattened)
        float *sol_array;
        //Array that contains the best solution vector and its fitness
        float *best_sol_fitness;
        //Array that contains the fitness of each solution
        float *fitness_array;
        int array_size;
        int sol_dim;
        float min;
        float max;
}abc_info_t;

//Device kernels
extern __device__ void scout_bee(curandState *state, float *sol, int dim, float min, float max);
extern __device__ void initialize_sol_array(curandState *state, float *sol, int sol_dim, float min, float max);
extern __device__ void print_float_sol(float *array, int size);

//Global kernels
extern __global__ void abc_algo(abc_info_t container);

#endif
