#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include "wb.h"
#include <iostream>
using namespace std;

/* Every thread gets exactly one value in the unsorted array. */
#define THREADS 512 // 2^9
#define BLOCKS 128 // 2^15
#define NUM_VALS THREADS*BLOCKS

void print_elapsed(clock_t start, clock_t stop)
{
  double elapsed = ((double) (stop - start)) / CLOCKS_PER_SEC;
  printf("Elapsed time: %.3fs\n", elapsed);
}

float random_float()
{
  return (float)rand()/(float)RAND_MAX;
}

void array_print(float *arr, int length) 
{
  int i;
  for (i = 0; i < length; ++i) {
    printf("%1.3f ",  arr[i]);
  }
  printf("\n");
}

void array_fill(float *arr, int length)
{
  srand(time(NULL));
  int i;
  for (i = 0; i < length; ++i) {
    arr[i] = random_float();
  }
}

__global__ void bitonic_sort_step(float *dev_values, int j, int k)
{
  unsigned int i, ixj;
  i = threadIdx.x + blockDim.x * blockIdx.x;
  ixj = i^j;

  if ((ixj)>i) {
    if ((i&k)==0) {
      if (dev_values[i]>dev_values[ixj]) {
        float temp = dev_values[i];
        dev_values[i] = dev_values[ixj];
        dev_values[ixj] = temp;
      }
    }
    if ((i&k)!=0) {
      if (dev_values[i]<dev_values[ixj]) {
        float temp = dev_values[i];
        dev_values[i] = dev_values[ixj];
        dev_values[ixj] = temp;
      }
    }
  }
}

void bitonic_sort(float *values)
{
  float *dev_values;
  size_t size = NUM_VALS * sizeof(float);

  cudaMalloc((void**) &dev_values, size);
  cudaMemcpy(dev_values, values, size, cudaMemcpyHostToDevice);

  dim3 blocks(BLOCKS,1);    
  dim3 threads(THREADS,1); 

  int j, k;

  for (k = 2; k <= NUM_VALS; k <<= 1) {
    for (j=k>>1; j>0; j=j>>1) {
      bitonic_sort_step<<<blocks, threads>>>(dev_values, j, k);
    }
  }
  cudaMemcpy(values, dev_values, size, cudaMemcpyDeviceToHost);
  cudaFree(dev_values);
}

int main(int argc, char **argv)
{
	wbArg_t args;
    args = wbArg_read(argc, argv);
  	clock_t start, stop;

  	int inputLength;
    float  *data;
    data = (float *)wbImport(wbArg_getInputFile(args, 0), &inputLength);
	
	cout<<"No: of Values : "<<inputLength<<endl;
	
	start = clock();
	bitonic_sort(data); 
	stop = clock();

	wbSolution(args, data, inputLength);
	print_elapsed(start, stop);
}
