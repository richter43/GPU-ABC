# Preface

Politecnico di Torino - 2022

An exam project for the GPU programming course.

Developed by Samuel Oreste Abreu - s281568 

# GPU-ABC

## Artificial Bee Colony algorithm implemented in a GPU.

### Execution instructions

The compilation step creates a ./bin/gpu and is executed as follows:

```
./gpu <Maximum number of iterations> <Ratio of onlooker to employed bees> <Maximum patience>
```

* Maximum number of iterations: number of times the ABC algorithm will be repeated.
* Ratio of onlooker to employed bees: self-explanatory, must be contained in the interval [0.0, 1.0].
* Maximum patience: amount of times the bee's solution is allowed to worsen before setting it to a scout bee.

**NOTE**: The desired function to be optimized has to be defined in the compute_fitness() function inside of the abc.cu file.

### Compilation instructions

**Compilation with debugging enabled**
```
make DEBUG=true
```

**Optimized code**
```
make gpu
```

**Cleaning the compilation folder**
```
make clean
```

### Code description

* main.cu
	* Main code, allocates the used memory and sets up the kernels to be run.
* abc.cu
	* API of the ABC algorithm.
* utils.cu
	* Utility functions.
* benchfuns.cu
	* Benchmark functions that are used for testing the ABC algorithm.
	* Changing the benchmark function constitutes in modifying the `FUNCTION` directive  inside of `abc.h`.

### Unit tests

* unittest.cu
	* Tests many of the defined functions/kernels.

### Main API functions

#### Host API

* `curandState *create_random_state(int blocks, int threads, int seed)`
	* Allocates memory and creates a cuRAND state inside of the device and returns its memory address.

* `struct abc_info_s`
	* Passed struct into the device for performing ABC computations

	```
	typedef struct abc_info_s{
			curandState *state;			// Array of the curandStates
			float *sol_array; 			// Array in which the solution will be stored
			float *best_sol_fitness; 	// Best solution and fitness found
			float *fitness_array; 		// Array which will contain the fitnesses
			int array_size; 			// Size of the array
			int sol_dim; 				// Dimension of the solution
			float min; 					// Minimum float
			float max; 					// Maximum float
			int num_iterations; 		// Maximum number of iterations
			int max_patience; 			// Maximum patience
			float ratio_ote; 			// Ratio of onlookers to employed bees
	}abc_info_t;
	```

#### Device API

* `void abc_algo(void)`
	* Kernel function that performs the ABC computation.

* `void setup_kernel(curandState *state, int seed)`
	* Initiatializes a curandState with the seed's argument.