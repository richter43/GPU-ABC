#include "benchfuns.h"

__host__ __device__ float rastrigin_1d(float x){
	return 10 + (powf(x, 2.0) - 10.0*cospif(2.0*x));
}
