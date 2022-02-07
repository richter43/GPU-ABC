#include <stdio.h>
#include <stdlib.h>
#include <helper_functions.h>
#include <helper_cuda.h>
#include <cuda.h>

#include "utils.h"

__device__ void scout_bee(curandState *state, float *sol, int dim, float min, float max){
	random_float_array(state, sol, dim, min, max);	
	return;
}
