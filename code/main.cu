#include <stdio.h>
#include <stdlib.h>
#include <helper_functions.h>
#include <helper_cuda.h>
#include <curand_kernel.h>

#include "main.h"
#include "abc.h"
#include "benchfuns.h"
#include "utils.h"

#define STORE_RESULTS 1
#define FILENAME "./values.csv"

//Bee state array

int main(int argc, char *argv[]){

	//Handling passed arguments
	if(argc != 4){
		fprintf(stderr, "Missing parameters (Iterations, ratio of onlooker to employed bees, max patience)\n");
		exit(EXIT_FAILURE);
	}

	int iterations;
	float ratio_ote;
	int max_patience;

	if(sscanf(argv[1], "%d", &iterations)!= 1 ||  sscanf(argv[2], "%f", &ratio_ote) != 1 || sscanf (argv[3], "%d", &max_patience) != 1){
                fprintf(stderr, "Error during parsing of the passed parameters\n");
		exit(EXIT_FAILURE);
        }

	if(ratio_ote < 0.0 || ratio_ote > 1.0){
		fprintf(stderr, "Ratio should be a number between 0.0 and 1.0\n");
		exit(EXIT_FAILURE);
	}

	//Host code
	float *h_best_sol_fitness; //Contains both best solution and its fitness
	h_best_sol_fitness = (float *) malloc_errorhandler(sizeof(float)*BLOCKS*(DIM+1));

	//Device code
	//Creation of a random state
	curandState *d_state = create_random_state(BLOCKS, THREADS, SEED);
	//Device memory allocation
	float *d_solutions, *d_best_sol_fitness;
	checkCudaErrors(cudaMalloc(&d_solutions, sizeof(float)*BLOCKS*THREADS*DIM));
	#if !SHARED_FITNESS
	float *d_fitness;
	checkCudaErrors(cudaMalloc(&d_fitness, sizeof(float)*BLOCKS*THREADS));
	#endif
	checkCudaErrors(cudaMalloc(&d_best_sol_fitness, sizeof(float)*BLOCKS*(DIM+1)));

	//Struct that contains all the relevant addresses and information
	#if SHARED_FITNESS
	abc_info_t h_container = { d_state, d_solutions, d_best_sol_fitness, BLOCKS*THREADS*DIM, DIM, MIN_FLOAT, MAX_FLOAT, iterations, max_patience, ratio_ote};
	#else
	abc_info_t h_container = { d_state, d_solutions, d_best_sol_fitness, d_fitness, BLOCKS*THREADS*DIM, DIM, MIN_FLOAT, MAX_FLOAT, iterations, max_patience, ratio_ote};
	#endif
	//Kernel execution
	#if TEST_CONSTANT
	copy_container_symbol(&h_container);
	abc_algo<<<BLOCKS,THREADS>>>();
	#else
	abc_algo<<<BLOCKS,THREADS>>>(h_container);
	#endif

	checkCudaErrors(cudaDeviceSynchronize());	
	checkCudaErrors(cudaMemcpy(h_best_sol_fitness, d_best_sol_fitness, sizeof(float)*BLOCKS*(DIM+1), cudaMemcpyDeviceToHost));

	//This is done for the sake of initializing the memory array to zero.
	float *tmp_sol = (float*) calloc(DIM,sizeof(float));
	float fitness_tmp_sum = 0.0;
	float max_tmp = 0.0;

	#if STORE_RESULTS
	FILE *fp = fopen(FILENAME, "a");
	if(fp == NULL){
		fprintf(stderr, "Could not create/open csv file.\n");
		exit(EXIT_FAILURE);
	}
	fprintf(fp, "hive,x,y,fitness\n");
	#endif

	if(tmp_sol == NULL){
		fprintf(stderr, "Could not allocate memory\n");
		exit(EXIT_FAILURE);
	}

	//Printing solutions
	for(int i = 0; i < BLOCKS; i++){
		printf("Hive %d solution: ", i);
		#if STORE_RESULTS
		fprintf(fp, "%d,", i);
		#endif
		for(int j = 0; j < DIM; j++){
			printf("%f ", h_best_sol_fitness[j + (DIM+1)*i]);
			#if STORE_RESULTS
			fprintf(fp, "%f,", h_best_sol_fitness[j + (DIM+1)*i]);
			#endif
		}
		
		if(max_tmp < h_best_sol_fitness[DIM + (DIM+1)*i]){
			max_tmp = h_best_sol_fitness[DIM + (DIM+1)*i];
		}
		printf("\nBest fitness: %f\n", h_best_sol_fitness[DIM + (DIM+1)*i]);
		#if STORE_RESULTS
		fprintf(fp, "%f\n", h_best_sol_fitness[DIM + (DIM+1)*i]);
		#endif
	}

	//Using the maximum to map the fitnesses to a range between 0 and 1 in order to avoid a potential overflow
	for(int i = 0; i < BLOCKS; i++){
		fitness_tmp_sum += h_best_sol_fitness[DIM + (DIM+1)*i]/max_tmp;
		h_best_sol_fitness[DIM + (DIM+1)*i] = h_best_sol_fitness[DIM + (DIM+1)*i]/max_tmp;
	}

	//Wisdom of crowds principle
	printf("Weighed wisdom of crowds solution: \n");
	for(int j = 0; j < DIM; j++){
		for(int i = 0; i < BLOCKS; i++){
			tmp_sol[j] += h_best_sol_fitness[j + (DIM+1)*i]*(h_best_sol_fitness[DIM + (DIM + 1)*i]/fitness_tmp_sum);
		}
		printf("%f ", tmp_sol[j]);
	}

	#if STORE_RESULTS
	fclose(fp);
	#endif

	free(h_best_sol_fitness);
	cudaFree(d_solutions);	
	cudaFree(d_state);	
	#if !SHARED_FITNESS
	cudaFree(d_fitness);
	#endif
	cudaFree(d_best_sol_fitness);

	return EXIT_SUCCESS;
}
