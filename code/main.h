#ifndef _H_MAIN
#define _H_MAIN

#define DEBUG 0

//This is temporary, after implementation it can be defined dynamically.
#define BLOCKS 1
#define THREADS 512
//Temporary problem dependent variables
#define SEED 0 //TODO: Set to zero when debugging
#define MIN_FLOAT -3.0
#define MAX_FLOAT 3.0
#define DIM 2
#define MAX_ITERATIONS 1024
#define MAX_PATIENCE 3
#define SHARED_ARRAYS 5
//Each thread behaves like a bee
#define TEST_CONSTANT 1

#endif
