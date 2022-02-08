#include <stdio.h>
#include <stdlib.h>
#include <helper_functions.h>
#include <helper_cuda.h>
#include <cuda.h>

#include "utils.h"
#include "benchfuns.h"
#include "abc.h"

#define DEBUG 1
#define STATIC_SHARED_FLOAT 128

__device__ void scout_bee(curandState *state, float *sol, int dim, float min, float max){
	random_float_array(state, sol, dim, min, max);	
	return;
}

__device__ void initialize_sol_array(curandState *state, float *sol, int sol_dim, float min, float max){
	random_float_array(state, sol, sol_dim, min, max);
	return;
}

__device__ void print_float_sol(float *array, int size){
	for(int i = 0; i < size; i++){
		printf("%f ", array[i]);
	}
	printf("\n");
}

__device__ float sum_fitness_naive(float *fitness_array, int size, int id){
	//TODO: sums the array that contains all of the fitness in a parallel manner (Alternated, log(n))
	//This should be per-block, thus, a maximum of MAX_THREADS bees are possible to spawn
	//A second version with shared memory of fitness can be created instead :)

	__shared__ float tmp_fitness[STATIC_SHARED_FLOAT];

	tmp_fitness[id] = fitness_array[id];

	for(int idx_div = 1; idx_div < size; idx_div *= 2){
		if(id % (2*idx_div) == 0 && id + idx_div < size){
			//Destructive sum
			tmp_fitness[id] += tmp_fitness[id+idx_div];
		}
	}

	return tmp_fitness[0];
}

__global__ void abc_algo(abc_info_t container){

	int id = threadIdx.x + blockDim.x*blockIdx.x;

	//Initialization step
	#if DEBUG
	if(id == 0){
		print_float_sol(&container.sol_array[id*container.sol_dim], container.sol_dim);
	}
	#endif
	initialize_sol_array(container.state, &container.sol_array[id*container.sol_dim], container.sol_dim, container.min, container.max);
	#if DEBUG	
	if(id == 0){
		print_float_sol(&container.sol_array[id*container.sol_dim], container.sol_dim);
	}
	#endif

	//Compute fitness
	//Function subject to change	
	rastrigin_nd(&container.fitness_array[id], &container.sol_array[id*container.sol_dim], container.sol_dim);

	#if DEBUG	
	if(id == 0){
		printf("Fitness: %f\n", container.fitness_array[id]);
	}
	#endif

	return;
}
