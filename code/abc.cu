#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <helper_functions.h>
#include <helper_cuda.h>
#include <cuda.h>

#include "main.h"
#include "utils.h"
#include "benchfuns.h"
#include "abc.h"

#define SUM_PROB 1
#define MAX_PROB 2
#define STATIC_SHARED THREADS
#define MAX_DIM DIM
#define PROB_TYPE SUM_PROB

#if TEST_CONSTANT
__constant__ abc_info_t container;

void copy_container_symbol(abc_info_t *src){
	cudaMemcpyToSymbol(container, src, sizeof(abc_info_t), 0, cudaMemcpyHostToDevice);
	return;
}
#endif

__device__ float *find_employed_bee_sol(float *sol_array, bee_status_t *bee_status_array, int sol_dim, int bee_id){
	//Finds the relative position of the selected bee's index

	int counter = 0;
	float *tmp_sol = sol_array;
	for(int i = 0; i < blockDim.x; i++){
		if(bee_status_array[i] == employed){
			if(bee_id == counter){
				if(i == threadIdx.x){
					//Adding one to the requested bee id so that it searches for the next employed bee.
					//In the event it doesn't find any it will return the first employed bee.
					bee_id +=1;
					continue;
				}
				tmp_sol = &sol_array[i*sol_dim];

				#if DEBUG
				printf("Global bee idx: %d Requested bee: %d Sol:", i + gridDim.x*blockDim.x, bee_id);
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
	//The original solution is mutated using a random value and an employed_sol array

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
	//Creates a random array and stores it into sol
	random_float_array(state, sol, sol_dim, min, max);
	return;
}

__device__ void print_float_sol(float *array, int size){
	//Prints the float array size

	for(int i = 0; i < size; i++){
		printf("%f ", array[i]);
	}
	printf("\n");
}

__device__ float sum_fitness_naive(float *fitness_array, int size, int id){
	//Sum computed in log(N)
	//Unused

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

__device__ void initialize_bee_status(curandState *state, bee_status_t *bee_status_array, int id, int *num_employed_bees){
	//Assigns task to a bee

	float val = curand_uniform(state);
	//Ratio is read from passed container
	if(val < container.ratio_ote){
		bee_status_array[id] = employed;
		atomicAdd(num_employed_bees, 1);
	}
	else{
		bee_status_array[id] = onlooker;
	}

	__syncthreads();

	#if DEBUG
	printf("Number of employed bees: %d\n", *num_employed_bees);
	#endif
	

	return;
}

__device__ void sum_fitness_employed_bees(float *sum, float fitness){
	//Adding atomically the current amount of employed bees

	atomicAdd(sum, fitness);
	return;
}

__device__ void max_fitness_employed_bees(float *max, float *fitness_array, int fitness_size,  bee_status_t *bee_status_array){
	//Finding the maximum among the employed bees
	
	for(int i = 0; i < fitness_size; i++){
		if(bee_status_array[i] == employed && *max < fitness_array[i]){
			*max = fitness_array[i];
		}
	}
	return;
}

__device__ bool optim_minimizer(float left, float right){
	//Minimizer function

	if(left > right){
		return true; 
	}
	return false;
}

__device__ bool optim_maximizer(float left, float right){
	//Maximizer function

	if(left < right){
		return true; 
	}
	return false;
}

__device__ bool best_fitness_naive(float *fitness_array, int size, int id, int *best_id, float *best_fitness){
	//Find which is the best solution in a parallel manner

	__shared__ int tmp_best_index[STATIC_SHARED];
	__shared__ float tmp_fitness[STATIC_SHARED];
	__shared__ bool tmp_return;

	if(threadIdx.x == 0){
		tmp_return = false;
	}

	tmp_best_index[threadIdx.x] = id;

	#if SHARED_FITNESS	
	tmp_fitness[threadIdx.x] = fitness_array[threadIdx.x];
	#else
	tmp_fitness[threadIdx.x] = fitness_array[id];
	#endif

	__syncthreads();

	for(int idx_div = 1; idx_div < size; idx_div *= 2){
		if(threadIdx.x % (2*idx_div) == 0 && threadIdx.x + idx_div < size){
			if(optim_maximizer(tmp_fitness[threadIdx.x], tmp_fitness[threadIdx.x+idx_div])){
				//Change id's current value if the optimization function returns true
				tmp_best_index[threadIdx.x] = tmp_best_index[threadIdx.x+idx_div];
				tmp_fitness[threadIdx.x] = tmp_fitness[threadIdx.x+idx_div];
			}
		}
	}

	if(threadIdx.x == 0 && optim_maximizer(*best_fitness, tmp_fitness[0])){
		*best_id = tmp_best_index[0];
		*best_fitness = tmp_fitness[0];
		tmp_return = true;
	}
	return tmp_return;
}

__device__ int select_random_employed_bee(curandState *state, bee_status_t *bee_status_array, int num_employed_bees){
	//Selects a local index of a random employed bee

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
	//Selects a random employed bee according to their probability density function

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
	//Self-explanatory, defined at compilation time

	float tmp_result;
	#if FUNCTION == RASTRIGIN
	rastrigin_nd(&tmp_result, input, dim);
	#elif FUNCTION == SPHERE
	sphere_nd(&tmp_result, input, dim);
	#elif FUNCTION == ROSENBROCK
	rosenbrock_nd(&tmp_result, input, dim);
	#endif
	#if FLIP_FUNCTION
	*result = 1/tmp_result;
	#endif
	return;
}

__device__ void employed_bee_handler(curandState *state, float *sol, bee_status_t *bee_status_array, int dim, float min, float max, int num_employed_bees, int id, float *fitness, int *patience, int max_patience, float *mutant_fitness){
	//Executes the behaviour of an employed bee

	//Selecting a random employed_bee
	int employed_bee_idx = select_random_employed_bee(state, bee_status_array, num_employed_bees);
	float *selected_employed_sol = find_employed_bee_sol(sol, bee_status_array, dim, employed_bee_idx);

	float mutant[MAX_DIM];
	mutate_solution(state, mutant, &sol[id*dim], selected_employed_sol, dim);
	//Compute mutant fitness
	compute_fitness(&mutant_fitness[id], mutant, dim);

	if(optim_maximizer(*fitness, mutant_fitness[id])){
		for(int i = 0; i < dim; i++){
			sol[id*dim + i] = mutant[i];
		}
		*fitness = mutant_fitness[id];
		*patience = 0;
	}
	else if(*patience > max_patience){
		__shared__ int num_employed_bees;
		atomicAdd(&num_employed_bees, -1);
                bee_status_array[id] = scout;
                *patience = 0;
        }      
        else{
                *patience += 1;
        }
}

__device__ void onlooker_bee_handler(curandState *state, float *sol, bee_status_t *bee_status_array, int dim, float min, float max, int num_employed_bees, int id, float *fitness, float *fitness_pdf, int *patience, int max_patience, float *mutant_fitness){
	//Executes the behaviour of an onlooker bee

	//Selecting an employed bee according to its fitness
	int employed_bee_idx = select_random_employed_bee_pdf(state, fitness_pdf, num_employed_bees);
	//Get mutation
	float *selected_employed_sol = find_employed_bee_sol(sol, bee_status_array, dim, employed_bee_idx);

	float mutant[MAX_DIM];
        mutate_solution(state, mutant, &sol[id*dim], selected_employed_sol, dim);
        //Compute mutant fitness
        compute_fitness(&mutant_fitness[id], mutant, dim);
 
        if(optim_maximizer(*fitness, mutant_fitness[id])){
                for(int i = 0; i < dim; i++){
                        sol[id*dim + i] = mutant[i];
                }
                *fitness = mutant_fitness[id];                
		*patience = 0;
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
	//Executes the behaviour of a scout bee

	random_float_array(state, sol, dim, min, max);	
	float old_fitness = *fitness;
	compute_fitness(fitness, sol, dim);
	if(optim_maximizer(old_fitness, *fitness)){
		*bee_status = onlooker;
	}
	else{
		*bee_status = employed;
	}
	return;
}

__device__ void adapt_fitness_array(float *prob_fitness, int num_employed_bees, bee_status_t *bee_status_array){
	//Put every computed fitness/pdf together

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

#if TEST_CONSTANT
__global__ void abc_algo(void){
#else
__global__ void abc_algo(abc_info_t container){
#endif
	//The kernel could also use shared memory for storing the fitness by setting SHARED_FITNESS to 1
	//Defining variables
	__shared__ int num_employed_bees;
	int id = threadIdx.x + blockDim.x*blockIdx.x;
	__shared__ int best_id;
	__shared__ float best_fitness;
	__shared__ bee_status_t bee_status_array[STATIC_SHARED];
	#if SHARED_FITNESS
	__shared__ float fitness[STATIC_SHARED];
	#endif
	__shared__ float scrap_array[STATIC_SHARED];

	int patience = 0; //Setting patience to an array of shared integers does not allow the profiling of the program (Crashes for some reason)

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
	//Prints the first bee's solution to check if it was randomly filled succesfully
	if(id == 0){
		print_float_sol(&container.sol_array[id*container.sol_dim], container.sol_dim);
	}
	#endif

	//Initializing bee's status
	initialize_bee_status(&container.state[id], bee_status_array, threadIdx.x, &num_employed_bees);

	//Compute fitness
	#if SHARED_FITNESS
	compute_fitness(&fitness[threadIdx.x], &container.sol_array[id*container.sol_dim], container.sol_dim);
	#else
	compute_fitness(&container.fitness_array[id], &container.sol_array[id*container.sol_dim], container.sol_dim);
	#endif
	#if PROB_TYPE == SUM_PROB
	__shared__ float sum_fitness;
	#elif PROB_TYPE == MAX_PROB
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

		__syncthreads();

		if(bee_status_array[threadIdx.x] == employed){	
			#if SHARED_FITNESS	
			employed_bee_handler(&container.state[id], &container.sol_array[blockIdx.x*blockDim.x*container.sol_dim], bee_status_array, container.sol_dim, container.min, container.max, num_employed_bees, threadIdx.x, &fitness[threadIdx.x], &patience, container.max_patience, scrap_array);
			#else
			employed_bee_handler(&container.state[id], &container.sol_array[blockIdx.x*blockDim.x*container.sol_dim], bee_status_array, container.sol_dim, container.min, container.max, num_employed_bees, threadIdx.x, &container.fitness_array[id], &patience, container.max_patience, scrap_array);
			#endif			

			#if PROB_TYPE == SUM_PROB			
			//Compute sum of the fitnesses
			
			#if SHARED_FITNESS	
			sum_fitness_employed_bees(&sum_fitness, fitness[threadIdx.x]);
			#else
			sum_fitness_employed_bees(&sum_fitness, container.fitness_array[id]);
			#endif			
			#if DEBUG
			printf("Sum fitness: %f\n", sum_fitness);
			#endif
			//Compute probability density

			#if SHARED_FITNESS	
			prob_fitness[threadIdx.x] = fitness[threadIdx.x]/sum_fitness;
			#else
			prob_fitness[threadIdx.x] = container.fitness_array[id]/sum_fitness;
			#endif
			#elif PROB_TYPE == MAX_PROB
			

			#if SHARED_FITNESS
			max_fitness_employed_bees(&max_fitness, fitness[threadIdx.x], blockDim.x, bee_status_array);
			prob_fitness[id] = 0.9*fitness[threadIdx.x]/max_fitness + 0.1;
			#else
			max_fitness_employed_bees(&max_fitness, container.fitness_array, blockDim.x, bee_status_array);
			prob_fitness[id] = 0.9*container.fitness_array[id]/max_fitness + 0.1;
			#endif
			#endif
		}
		__syncthreads();
		if(threadIdx.x == 0){
			//Adapt probability density array
			adapt_fitness_array(prob_fitness, num_employed_bees, bee_status_array);
		}
		__syncthreads();

		if(bee_status_array[threadIdx.x] == onlooker){
			#if SHARED_FITNESS
			onlooker_bee_handler(&container.state[id], &container.sol_array[blockIdx.x*blockDim.x*container.sol_dim], bee_status_array, container.sol_dim, container.min, container.max, num_employed_bees, threadIdx.x, &fitness[threadIdx.x], prob_fitness, &patience, container.max_patience, scrap_array);
			#else
			onlooker_bee_handler(&container.state[id], &container.sol_array[blockIdx.x*blockDim.x*container.sol_dim], bee_status_array, container.sol_dim, container.min, container.max, num_employed_bees, threadIdx.x, &container.fitness_array[id], prob_fitness, &patience, container.max_patience, scrap_array);
			#endif
		}

		__syncthreads();

		if(bee_status_array[threadIdx.x] == scout){
			#if SHARED_FITNESS
			scout_bee_handler(&container.state[id], &container.sol_array[id*container.sol_dim], &bee_status_array[threadIdx.x], &fitness[threadIdx.x], container.sol_dim, container.min, container.max);
			#else
			scout_bee_handler(&container.state[id], &container.sol_array[id*container.sol_dim], &bee_status_array[threadIdx.x], &container.fitness_array[id], container.sol_dim, container.min, container.max);
			#endif
		}

		#if DEBUG
		//Checks if the fitness was computed correctly
		if(id == 0){
			printf("Blockdim: %d\n", blockDim.x);
			for(int i = 0; i < blockDim.x; i++){
				printf("Thread %d -> Sol: ", i);
				print_float_sol(&container.sol_array[i*container.sol_dim], container.sol_dim);
				#if SHARED_FITNESS
				printf("Fitness: %f\n", container.fitness_array[i]);
				#else
				printf("Fitness: %f\n", fitness[threadIdx.x]);
				#endif
			}
		}
		#endif

		__syncthreads();

		//Find best fitness and stores its solution in global memory
		#if SHARED_FITNESS	
		if(best_fitness_naive(fitness, blockDim.x, id, &best_id, &best_fitness)){
		#else
		if(best_fitness_naive(container.fitness_array, blockDim.x, id, &best_id, &best_fitness)){
		#endif
			if(threadIdx.x < container.sol_dim){
				container.best_sol_fitness[threadIdx.x + (container.sol_dim + 1)*blockIdx.x] = container.sol_array[best_id*container.sol_dim + threadIdx.x]; 
			}
			if(threadIdx.x == 0){
				container.best_sol_fitness[container.sol_dim + (container.sol_dim + 1)*blockIdx.x] = best_fitness;
				#if DEBUG
				printf("Best ID: %d\nBest Fitness: %f\n", best_id, best_fitness);
				#endif
			}
		}

	}
	
	return;
}
