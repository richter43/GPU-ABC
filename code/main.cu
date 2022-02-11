#include <stdio.h>
#include <stdlib.h>
#include <helper_functions.h>
#include <helper_cuda.h>
#include <curand_kernel.h>

#include "abc.h"
#include "benchfuns.h"
#include "utils.h"

#define DEBUG 1

//This is temporary, after implementation it can be defined dynamically.
#define BLOCKS 1
#define THREADS 32
//Temporary problem dependent variables
#define SEED 0 //TODO: Set to zero when debugging
#define MIN_FLOAT -3.0
#define MAX_FLOAT 3.0
#define DIM 2
#define MAX_ITERATIONS 128
//Each thread behaves like a bee

//Bee state array

int main(void){

	//Host code
	float *h_best_sol_fitness; //Contains both best solution and its fitness
	h_best_sol_fitness = (float *) malloc_errorhandler(sizeof(float)*(DIM+1));

	//Device code
	//Creation of a random state
	curandState *d_state = create_random_state(BLOCKS, THREADS, SEED);
	//Device memory allocation
	float *d_solutions, *d_fitness, *d_best_sol_fitness;
	checkCudaErrors(cudaMalloc(&d_solutions, sizeof(float)*BLOCKS*THREADS*DIM));
	checkCudaErrors(cudaMalloc(&d_fitness, sizeof(float)*BLOCKS*THREADS));
	checkCudaErrors(cudaMalloc(&d_best_sol_fitness, sizeof(float)*(DIM+1)));

	//Struct that contains all the relevant addresses and information
	abc_info_t container = { d_state, d_solutions, d_best_sol_fitness, d_fitness, BLOCKS*THREADS*DIM, DIM, MIN_FLOAT, MAX_FLOAT, MAX_ITERATIONS};
	//Kernel execution
	abc_algo<<<BLOCKS,THREADS>>>(container);
	
	checkCudaErrors(cudaMemcpy(h_best_sol_fitness, d_best_sol_fitness, sizeof(float)*(DIM+1), cudaMemcpyDeviceToHost));

	free(h_best_sol_fitness);
	cudaFree(d_solutions);	
	cudaFree(d_state);	
	cudaFree(d_fitness);
	cudaFree(d_best_sol_fitness);

	return EXIT_SUCCESS;
}
