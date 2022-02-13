#ifndef _H_ABC
#define _H_ABC

#define RASTRIGIN 1
#define SPHERE 2
#define ROSENBROCK 3
#define FUNCTION SPHERE
#define FLIP_FUNCTION 1
#define SHARED_FITNESS 0

//enum
typedef enum BEE_STATUS
{
	scout,
	employed,
	onlooker
}bee_status_t;

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
	int num_iterations;
	int max_patience;
}abc_info_t;

//Utility function
#if TEST_CONSTANT
extern void copy_container_symbol(abc_info_t *src);
#endif

//Device kernels
extern __device__ void scout_bee(curandState *state, float *sol, int dim, float min, float max);
extern __device__ void initialize_sol_array(curandState *state, float *sol, int sol_dim, float min, float max);
extern __device__ void print_float_sol(float *array, int size);
extern __device__ float sum_fitness_naive(float *fitness_array, int size, int id);
//Global kernels
#if TEST_CONSTANT
extern __global__ void abc_algo(void);
#else
extern __global__ void abc_algo(abc_info_t container);
#endif

#endif
