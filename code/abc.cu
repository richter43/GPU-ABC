#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <helper_functions.h>
#include <helper_cuda.h>
#include <cuda.h>

#include "utils.h"
#include "benchfuns.h"
#include "abc.h"

#define DEBUG 1
#define SUM_PROB 1
#define MAX_PROB 0
#define STATIC_SHARED 128

__device__ float *find_employed_bee_sol(float *sol_array, bee_status_t *bee_status_array, int sol_dim, int bee_id, int thread_id){
	int counter = 0;
	float *tmp_sol = sol_array;
	for(int i = 0; i < blockDim.x; i++){
		if(bee_status_array[i] == employed){
			if(bee_id == counter){
				if(i == thread_id){
					bee_id +=1;
					continue;
				}
				tmp_sol = &sol_array[i*sol_dim];

				#if DEBUG
				printf("Global bee idx: %d Requested bee: %d Sol:", i, bee_id);
				print_float_sol(tmp_sol, sol_dim);
				#endif

				break;
			}
			else{
				counter += 1;
			}
		}
	}


	return tmp_sol;
}

__device__ void mutate_solution(curandState *state, float *mutant, float *original, float *employed_sol, int size){
	
	float random_float;

	for(int i = 0; i < size; i++){
		random_float = get_random_float(state, -1, 1);
		mutant[i] = original[i] + random_float*(original[i] - employed_sol[i]);

		#if DEBUG
		printf("Mutation:%f Orig:%f Employed:%f Random float:%f\n", mutant[i], original[i], employed_sol[i], random_float);
		#endif
	}


	//Compute fitness and compare
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
	//Dynamic shared memory could also be introduced

	__shared__ float tmp_fitness[STATIC_SHARED];

	tmp_fitness[id] = fitness_array[id];

	for(int idx_div = 1; idx_div < size; idx_div *= 2){
		if(id % (2*idx_div) == 0 && id + idx_div < size){
			//Destructive sum
			tmp_fitness[id] += tmp_fitness[id+idx_div];
		}
	}

	return tmp_fitness[0];
}

__device__ int initialize_bee_status(curandState *state, bee_status_t *bee_status_array, int id){
	
	float val = curand_uniform(state);
	__shared__ int num_employed_bees;
	//The ratio may change
	if(val > 0.5){
		bee_status_array[id] = employed;
		atomicAdd(&num_employed_bees, 1);
	}
	else{
		bee_status_array[id] = onlooker;
	}

	#if DEBUG
	printf("Number of employed bees: %d\n", num_employed_bees);
	#endif
	

	return num_employed_bees;
}

__device__ void sum_fitness_employed_bees(float *sum, float fitness){
	
	atomicAdd(sum, fitness);
	return;
}

__device__ void max_fitness_employed_bees(float *max, float *fitness_array, int fitness_size,  bee_status_t *bee_status_array){
	
	for(int i = 0; i < fitness_size; i++){
		if(bee_status_array[i] == employed && *max < fitness_array[i]){
			*max = fitness_array[i];
		}
	}
	return;
}

__device__ bool optim_minimizer(float left, float right){

	if(left > right){
		return true; 
	}
	return false;
}

__device__ bool optim_maximizer(float left, float right){

	if(left < right){
		return true; 
	}
	return false;
}

__device__ bool best_fitness_naive(float *fitness_array, int size, int id, int *best_id, float *best_fitness){
	__shared__ int tmp_best_index[STATIC_SHARED];
	__shared__ float tmp_fitness[STATIC_SHARED];
	__shared__ bool tmp_return;

	if(id == 0){
		tmp_return = false;
	}

	tmp_best_index[id] = id;
	tmp_fitness[id] = fitness_array[id];

	for(int idx_div = 1; idx_div < size; idx_div *= 2){
		if(id % (2*idx_div) == 0 && id + idx_div < size){
			if(optim_maximizer(tmp_fitness[id], tmp_fitness[id+idx_div])){
				//Change id's current value if the optimization function returns true
				tmp_best_index[id] = tmp_best_index[id+idx_div];
				tmp_fitness[id] = tmp_fitness[id+idx_div];
			}
		}
	}

	if(id == 0 && optim_maximizer(*best_fitness, tmp_fitness[0])){
		*best_id = tmp_best_index[0];
		*best_fitness = tmp_fitness[0];
		tmp_return = true;
	}
	return tmp_return;
}

//Diff pdf -> add all pdfs until > random value then that's the idx :)

__device__ int select_random_employed_bee(curandState *state, bee_status_t *bee_status_array, int num_employed_bees){
	int employed_bee_idx = curand(state)%num_employed_bees;
	if(employed_bee_idx < 0){
		//Remainder could be negative due to unsigned int
		employed_bee_idx *= -1;
	}
	#if DEBUG
	printf("Selected bee %d\n", employed_bee_idx);
	#endif
	return employed_bee_idx;
}

__device__ int select_random_employed_bee_pdf(curandState *state, float *fitness_pdf, int num_employed_bees){
	
	float tmp_rand = curand_uniform(state);
	float tmp_sum = 0.0;
	int counter;

	for(counter = 0; counter < num_employed_bees && tmp_sum < tmp_rand; counter++){
		tmp_sum += fitness_pdf[counter];
	}

	counter -= 1;

	#if DEBUG
	printf("Probability:%f Selected bee: %d\n", tmp_rand, counter);
	#endif
	
	return counter;

}

__device__ void compute_fitness(float *result, float *input, int dim){
	float tmp_result;
	rastrigin_nd(&tmp_result, input, dim);
	*result = 1/tmp_result;
	return;
}

__device__ void employed_bee_handler(curandState *state, float *sol, bee_status_t *bee_status_array, int dim, float min, float max, int num_employed_bees, int id, float *fitness, int *patience, int max_patience){
	//Selecting a random employed_bee
	int employed_bee_idx = select_random_employed_bee(state, bee_status_array, num_employed_bees);
	float *selected_employed_sol = find_employed_bee_sol(sol, bee_status_array, dim, employed_bee_idx, id);

	float mutant[STATIC_SHARED];
	mutate_solution(state, mutant, &sol[id*dim], selected_employed_sol, dim);
	//Compute utant fitness
	float mutant_fitness;
	compute_fitness(&mutant_fitness, mutant, dim);

	if(optim_maximizer(*fitness, mutant_fitness)){
		for(int i = 0; i < dim; i++){
			sol[id*dim + i] = mutant[i];
		}
		*fitness = mutant_fitness;
	}
	else if(*patience > max_patience){
                bee_status_array[id] = scout;
                *patience = 0;
        }      
        else{
                *patience += 1;
        }
	//Requires a patience counter

}

__device__ void onlooker_bee_handler(curandState *state, float *sol, bee_status_t *bee_status_array, int dim, float min, float max, int num_employed_bees, int id, float *fitness, float *fitness_pdf, int *patience, int max_patience){
	
	//Selecting an employed bee according to its fitness
	int employed_bee_idx = select_random_employed_bee_pdf(state, fitness_pdf, num_employed_bees);
	//Get mutation
	float *selected_employed_sol = find_employed_bee_sol(sol, bee_status_array, dim, employed_bee_idx, id);

	float mutant[STATIC_SHARED];
	mutate_solution(state, mutant, &sol[id*dim], selected_employed_sol, dim);
	//Compute mutant fitness
	float mutant_fitness;
	compute_fitness(&mutant_fitness, mutant, dim);
	
	if(optim_maximizer(*fitness, mutant_fitness)){
		for(int i = 0; i < dim; i++){
			sol[id*dim + i] = mutant[i];
		}
		*fitness = mutant_fitness;
	}
	else if(*patience > max_patience){
		bee_status_array[id] = scout;
		*patience = 0;
	}
	else{
		*patience += 1;
	}
	return;
}

__device__ void scout_bee_handler(curandState *state, float *sol, bee_status_t *bee_status, float *fitness, int dim, float min, float max){
	random_float_array(state, sol, dim, min, max);	
	compute_fitness(fitness, sol, dim);
	*bee_status = onlooker;
	return;
}

__device__ void adapt_fitness_array(float *prob_fitness, int num_employed_bees, bee_status_t *bee_status_array){

	int counter = 0;

	#if DEBUG
	printf("Fitness and index: ");
	#endif

	for(int i = 0; i < blockDim.x; i++){
		if(bee_status_array[i] == employed){
			prob_fitness[counter] = prob_fitness[i];
			#if DEBUG
			printf("%d %f\n", i, prob_fitness[counter]);
			#endif
			counter += 1;
		}
		
	}

	return;
}

__global__ void abc_algo(abc_info_t container){
	//The kernel could also use shared memory for storing the fitness
	//Shared mem size = dim (Storing) + dim (temp sum_fitness)

	//Defining variables
	int id = threadIdx.x + blockDim.x*blockIdx.x;
	__shared__ int best_id;
	__shared__ float best_fitness;
	__shared__ bee_status_t bee_status_array[STATIC_SHARED];
	int patience = 0;

	//Initialization step
	#if DEBUG
	//Prints the first bee's solution to check if it's empty
	if(id == 0){
		print_float_sol(&container.sol_array[id*container.sol_dim], container.sol_dim);
	}
	#endif
	
	//Initializing solutions
	initialize_sol_array(&container.state[id], &container.sol_array[id*container.sol_dim], container.sol_dim, container.min, container.max);

	#if DEBUG	
	//Prints the first bee's solution to check if it was randomly filled
	if(id == 0){
		print_float_sol(&container.sol_array[id*container.sol_dim], container.sol_dim);
	}
	#endif

	//Initializing bee's status
	int num_employed_bees = initialize_bee_status(&container.state[id], bee_status_array, id);

	//Compute fitness
	compute_fitness(&container.fitness_array[id], &container.sol_array[id*container.sol_dim], container.sol_dim);
	#if SUM_PROB
	__shared__ float sum_fitness;
	#elif MAX_PROB
	__shared__ float max_fitness;
	#endif
	__shared__ float prob_fitness[STATIC_SHARED];


	#if DEBUG
	if(id == 0){
		for(int i = 0; i < blockDim.x; i++){
			printf("Thread:%d\nBee status: %d\n", i, bee_status_array[i]);
		}
	}
	#endif

	for(int i = 0; i < container.num_iterations; i++){
		if(bee_status_array[id] == employed){	
			employed_bee_handler(&container.state[id], container.sol_array, bee_status_array, container.sol_dim, container.min, container.max, num_employed_bees, id, &container.fitness_array[id], &patience, container.max_patience);
			#if SUM_PROB			
			//Compute sum of the fitnesses
			sum_fitness_employed_bees(&sum_fitness, container.fitness_array[id]);
			#if DEBUG
			printf("Sum fitness: %f\n", sum_fitness);
			#endif
			//Compute probability density
			prob_fitness[id] = container.fitness_array[id]/sum_fitness;
			#elif MAX_PROB
			
			max_fitness_employed_bees(&max_fitness, container.fitness_array, blockDim.x, bee_status_array);
			prob_fitness[id] = 0.9*container.fitness_array[id]/max_fitness + 0.1;
			
			#endif
		}

		if(id == 0){
			//Adapt probability density array
			adapt_fitness_array(prob_fitness, num_employed_bees, bee_status_array);
		}

		if(bee_status_array[id] == onlooker){
			onlooker_bee_handler(&container.state[id], container.sol_array, bee_status_array, container.sol_dim, container.min, container.max, num_employed_bees, id, &container.fitness_array[id], prob_fitness, &patience, container.max_patience);
		}
		else{
			scout_bee_handler(&container.state[id], &container.sol_array[id*container.sol_dim], &bee_status_array[id], &container.fitness_array[id], container.sol_dim, container.min, container.max);
		}

		#if DEBUG
		//Checks if the fitness was computed correctly
		if(id == 0){
			printf("Blockdim: %d\n", blockDim.x);
			for(int i = 0; i < blockDim.x; i++){
				printf("Thread %d -> Sol: ", i);
				print_float_sol(&container.sol_array[i*container.sol_dim], container.sol_dim);
				printf("Fitness: %f\n", container.fitness_array[i]);
			}
		}
		#endif

		if(best_fitness_naive(container.fitness_array, blockDim.x, id, &best_id, &best_fitness)){
			if(id < container.sol_dim){
				container.best_sol_fitness[id] = container.sol_array[best_id*container.sol_dim + id]; 
			}
			if(id == 0){
				container.best_sol_fitness[container.sol_dim] = best_fitness;
				#if DEBUG
				printf("Best ID: %d\nBest Fitness: %f\n", best_id, best_fitness);
				#endif
			}
		}

	}

	return;
}
