#include <stdio.h>
#include <stdlib.h>
#include <helper_functions.h>
#include <helper_cuda.h>
#include <cuda.h>

#include "utils.h"
#include "benchfuns.h"

#define DEBUG 1

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

__device__ void sum_fitness(void){
	//TODO: sums the array that contains all of the fitness in a parallel manner (Alternated, log(n))
	//This should be per-block, thus, a maximum of MAX_THREADS bees are possible to spawn
	
	return;
}

__global__ void abc_algo(curandState *state, float *sol_array, int array_size, int sol_dim, float min, float max){

	int id = threadIdx.x + blockDim.x*blockIdx.x;

	//Initialization step
	#if DEBUG
	if(id == 0){
		print_float_sol(&sol_array[id*sol_dim], sol_dim);
	}
	#endif
	initialize_sol_array(state, &sol_array[id*sol_dim], sol_dim, min, max);
	#if DEBUG	
	if(id == 0){
		print_float_sol(&sol_array[id*sol_dim], sol_dim);
	}
	#endif

	//Compute fitness

	float fitness;
	//Function subject to change	
	rastrigin_nd(&fitness, &sol_array[id*sol_dim], sol_dim);

	#if DEBUG	
	if(id == 0){
		printf("Fitness: %f\n", fitness);
	}
	#endif

	return;
}
