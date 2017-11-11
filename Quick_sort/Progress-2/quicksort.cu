#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "wb.h"

#define TEST_SIZE 5000 
#define RAND_RANGE 10000

 void printArr( int arr[], int n )
{
    int i;
    for ( i = 0; i < n; ++i )
        printf( "%d ", arr[i] );
}
__device__ int d_size;

__global__ void partition (int *arr, int *arr_l, int *arr_h, int n)
{
    int z = blockIdx.x*blockDim.x+threadIdx.x;
    d_size = 0;
    __syncthreads();
    if (z<n)
      {
        int h = arr_h[z];
        int l = arr_l[z];
        int x = arr[h];
        int i = (l - 1);
        int temp;
        for (int j = l; j <= h- 1; j++)
          {
            if (arr[j] <= x)
              {
                i++;
                temp = arr[i];
                arr[i] = arr[j];
                arr[j] = temp;
              }
          }
        temp = arr[i+1];
        arr[i+1] = arr[h];
        arr[h] = temp;
        int p = (i + 1);
        if (p-1 > l)
          {
            int ind = atomicAdd(&d_size, 1);
            arr_l[ind] = l;
            arr_h[ind] = p-1;  
          }
        if ( p+1 < h )
          {
            int ind = atomicAdd(&d_size, 1);
            arr_l[ind] = p+1;
            arr_h[ind] = h; 
          }
      }
}
 
float * quickSortIterative (float *arr, int l, int h)
{
    float lstack[ h - l + 1 ], hstack[ h - l + 1];
 
    int top = -1, *d_d, *d_l, *d_h;
 
    lstack[ ++top ] = l;
    hstack[ top ] = h;

    cudaMalloc(&d_d, (h-l+1)*sizeof(int));
    cudaMemcpy(d_d, arr,(h-l+1)*sizeof(int),cudaMemcpyHostToDevice);

    cudaMalloc(&d_l, (h-l+1)*sizeof(int));
    cudaMemcpy(d_l, lstack,(h-l+1)*sizeof(int),cudaMemcpyHostToDevice);

    cudaMalloc(&d_h, (h-l+1)*sizeof(int));
    cudaMemcpy(d_h, hstack,(h-l+1)*sizeof(int),cudaMemcpyHostToDevice);
    float n_t = 1;
    float n_b = 1;
    float n_i = 1; 
    while ( n_i > 0 )
    {
        partition<<<n_b,n_t>>>( d_d, d_l, d_h, n_i);
        float answer;
        cudaMemcpyFromSymbol(&answer, d_size, sizeof(int), 0, cudaMemcpyDeviceToHost); 
        if (answer < 1024)
          {
            n_t = answer;
          }
        else
          {
            n_t = 1024;
            n_b = (answer/n_t + (answer%n_t==0?0:1));
          }
        n_i = answer;
        cudaMemcpy(arr, d_d,(h-l+1)*sizeof(int),cudaMemcpyDeviceToHost);
    }
    return arr;
}
 

 
int main(int argc, char **argv) {

	wbArg_t args;
   	args = wbArg_read(argc, argv);
   	
    	float *h_inVals;
    	float *h_outVals;
      	size_t memsize = sizeof(float) * TEST_SIZE;
      	int numElements=TEST_SIZE;
      	
       	wbTime_start(Generic, "Importing data and creating memory on host");
    	h_inVals = (float *) wbImport(wbArg_getInputFile(args, 0), &numElements);
     	h_outVals = (float *)malloc(memsize);
      	wbTime_stop(Generic, "Importing data and creating memory on host");
       	
       	wbLog(TRACE, "The number of input elements in the input is ", numElements);
      
        //unsigned int *d_inputVals;
    	//unsigned int *d_outputVals;
    	 /*      
        wbTime_start(GPU, "Allocating GPU memory.");
    	cudaMalloc(&d_inputVals, memsize);
    	cudaMalloc(&d_outputVals, memsize);
    	wbTime_stop(GPU, "Allocating GPU memory.");
    	
    	wbTime_start(GPU, "Copying input memory to the GPU.");
    	cudaMemcpy(d_inputVals, h_inVals, memsize, cudaMemcpyHostToDevice);
    	wbTime_stop(GPU, "Copying input memory to the GPU.");
    	*/
    	
    	 wbTime_start(Compute, "Performing CUDA computation");
	 h_outVals=quickSortIterative( h_inVals, 0, TEST_SIZE - 1 );
	 wbTime_stop(Compute, "Performing CUDA computation");
	 
	/* wbTime_start(Copy, "Copying output memory to the CPU");
 	 cudaMemcpy(h_outVals, d_outputVals, memsize, cudaMemcpyDeviceToHost);
    	 wbTime_stop(Copy, "Copying output memory to the CPU");
    	 
    	  wbTime_start(GPU, "Freeing GPU Memory");
    	  cudaFree(d_inputVals);
    	  cudaFree(d_inputPos);
    	  cudaFree(d_outputVals);
    	  cudaFree(d_outputPos);
    	  wbTime_stop(GPU, "Freeing GPU Memory");
    	  */
    	   wbSolution(args, h_outVals, TEST_SIZE);
    //instead of quickSort function define a new function 
    //quickSortIterative( arr, 0, n - 1 );
    //printArr( arr, n );
    return 0;
}
