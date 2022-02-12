#include <stdio.h>
#include <stdlib.h>
#include <helper_functions.h>
#include <helper_cuda.h>
#include <curand_kernel.h>

#include "main.h"
#include "abc.h"
#include "benchfuns.h"
#include "utils.h"

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
	abc_info_t h_container = { d_state, d_solutions, d_best_sol_fitness, d_fitness, BLOCKS*THREADS*DIM, DIM, MIN_FLOAT, MAX_FLOAT, MAX_ITERATIONS, MAX_PATIENCE};
	//Kernel execution
	#if TEST_CONSTANT
	copy_container_symbol(&h_container);
	abc_algo<<<BLOCKS,THREADS>>>();
	#else
	abc_algo<<<BLOCKS,THREADS>>>(h_container);
	#endif

	checkCudaErrors(cudaDeviceSynchronize());
	
	checkCudaErrors(cudaMemcpy(h_best_sol_fitness, d_best_sol_fitness, sizeof(float)*(DIM+1), cudaMemcpyDeviceToHost));

	printf("Best solution: ");
	for(int i = 0; i < DIM; i++){
		printf("%f ", h_best_sol_fitness[i]);
	}
	printf("\nBest fitness: %f\n", h_best_sol_fitness[DIM]);

	free(h_best_sol_fitness);
	cudaFree(d_solutions);	
	cudaFree(d_state);	
	cudaFree(d_fitness);
	cudaFree(d_best_sol_fitness);

	return EXIT_SUCCESS;
}
