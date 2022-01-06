#include <stdio.h>
#include <stdlib.h>
#include <helper_functions.h>
#include <helper_cuda.h>

#define MASK_HW 3
#define PADDING 2
#define DEBUG 0

#include "utils.h"
#include "test.h"

__global__ void convolve2d(char *I, char *R, float* __restrict__ M, int height, int width);

int main(void){

	//Host code

	//Gaussian blur in matrix form
	float **g_blur = gaussianblur_matrix(1,MASK_HW);
	//Gaussian blur in flat form
	float *flat_blur = (float*) unroll_matrix((void **)g_blur, MASK_HW, MASK_HW, sizeof(float), 0);
	//Free gaussian blur matrix
	free_matrix((void**)g_blur, MASK_HW);
	//Open image into memory
	t_pgm_image t_image = open_pgm("./images/Vista_Teatro_teresa_carreño.pgm");
	//Flatten image	
	char *flat_image = (char*) unroll_matrix((void**)t_image.image, \
			t_image.height, t_image.width, sizeof(char), PADDING);
	//End host code

	//Device code
	//Declaring variables
	float *d_flat_blur, *d_image, *d_result;
	//Allocating memory in device
	checkCudaErrors(cudaMalloc(&d_flat_blur, sizeof(float)*MASK_HW*MASK_HW));
	checkCudaErrors(cudaMalloc(&d_image, sizeof(char)*(t_image.height+2*PADDING)*(t_image.width+2*PADDING)));
	checkCudaErrors(cudaMalloc(&d_result, sizeof(char)*t_image.height*t_image.width));
	//Copying memory to device
	checkCudaErrors(cudaMemcpy(d_flat_blur, flat_blur, sizeof(float)*MASK_HW*MASK_HW, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_image, flat_image, \
				sizeof(char)*(t_image.height+2*PADDING)*(t_image.width+2*PADDING), \
				cudaMemcpyHostToDevice));
	//End device code	

	#if DEBUG
	d_print_matrix<<<1,MASK_HW*MASK_HW>>>(d_flat_test);
	#endif

	dim3 blocks = {t_image.height, t_image.width,1};
	dim3 threads = {1024,1024,1};

	convolve2d<blocks, threads>(d_image, d_result, d_flat_blur);

	cudaFree(d_flat_blur);
	cudaFree(d_image);
	cudaFree(d_result);

	free(flat_blur);
	free(flat_image);
	free_matrix((void**) t_image.image, t_image.height);

	return EXIT_SUCCESS;
}


__global__ void convolve2d(char *I, char *R, float* __restrict__ M, int height, int width){
	
	int tx = threadIdx.x + blockDim.x * blockIdx.x;
	int ty = threadIdx.y + blockDim.y * blockIdx.y;

	for(int i = 0; i < MASK_HW; i++){
		for(int j = 0; j < MASK_HW; i++){
			
		}
	}

	return;	
}
