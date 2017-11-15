#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include "wb.h"
#include <bits/stdc++.h>
using namespace std;

#define TEST_SIZE 50000 
#define RAND_RANGE 5000
#define BLOCK_WIDTH 32 
#define CEILING_DIVIDE(X, Y) (1 + (((X) - 1) / (Y)))

__global__ void partialScan(unsigned int *d_in,
                            unsigned int *d_out,
                            unsigned int *d_total,
                            size_t n)
{
    __shared__ unsigned int temp[BLOCK_WIDTH];
    int tx = threadIdx.x;
    int bx = blockIdx.x;
    int index = BLOCK_WIDTH * bx + tx;

    if(index < n) {
        temp[tx] = d_in[index];
    } else { temp[tx] = 0; }
    __syncthreads();

   
    for(int offset = 1; offset < BLOCK_WIDTH; offset <<= 1) {
        if(tx + offset < BLOCK_WIDTH) {
            temp[tx + offset] += temp[tx];
        }
        __syncthreads();
    }

    
    if(tx +1 < BLOCK_WIDTH && index + 1 < n) {
        d_out[index + 1] = temp[tx];
    }
    d_out[0] = 0;

    
    d_total[bx] = temp[BLOCK_WIDTH - 1];
}


__global__ void mapScan(unsigned int *d_array, unsigned int *d_total, size_t n) {
    int tx = threadIdx.x;
    int bx = blockIdx.x;
    int index = BLOCK_WIDTH * bx + tx;

    if(index < n) {
        d_array[index] += d_total[bx];
    }
}


__global__ void mapPredicate(unsigned int *d_zeros,
                             unsigned int *d_ones,
                             unsigned int *d_in,
                             unsigned int bit,
                             size_t n)
{
    int tx = threadIdx.x;
    int bx = blockIdx.x;
    int index = BLOCK_WIDTH * bx + tx;

    if(index < n) {
        unsigned int isOne = (d_in[index] >> bit) & 1;
        d_ones[index] = isOne;
        d_zeros[index] = 1 - isOne;
    }
}


__global__ void scatter(unsigned int *d_inVals,
                        unsigned int *d_outVals,
                        unsigned int *d_inPos,
                        unsigned int *d_outPos,
                        unsigned int *d_zerosScan,
                        unsigned int *d_onesScan,
                        unsigned int *d_zerosPredicate,
                        unsigned int *d_onesPredicate,
                        size_t n)
{
    int tx = threadIdx.x;
    int bx = blockIdx.x;
    int index = BLOCK_WIDTH * bx + tx;
    int offset = d_zerosScan[n - 1] + d_zerosPredicate[n - 1];

    if(index < n) {
        int scatterIdx;
        if(d_zerosPredicate[index]) {
            scatterIdx = d_zerosScan[index];
        } else {
            scatterIdx = d_onesScan[index] + offset;
        }
        if(scatterIdx < n) { 
            d_outVals[scatterIdx] = d_inVals[index];
            d_outPos[scatterIdx] = d_inPos[index];
        }
    }
}

void totalScan(unsigned int *d_in, unsigned int *d_out, size_t n) {
    size_t numBlocks = CEILING_DIVIDE(n, BLOCK_WIDTH);
    unsigned int *d_total;
    cudaMalloc(&d_total, sizeof(unsigned int) * numBlocks);
    cudaMemset(d_total, 0, sizeof(unsigned int) * numBlocks);

    partialScan<<<numBlocks, BLOCK_WIDTH>>>(d_in, d_out, d_total, n);

    if(numBlocks > 1) {
        unsigned int *d_total_scanned;
        cudaMalloc(&d_total_scanned, sizeof(unsigned int) * numBlocks);
        cudaMemset(d_total_scanned, 0, sizeof(unsigned int) * numBlocks);

        totalScan(d_total, d_total_scanned, numBlocks);

        mapScan<<<numBlocks, BLOCK_WIDTH>>>(d_out, d_total_scanned, n);

        cudaFree(d_total_scanned);
    }

    cudaFree(d_total);
}

void radix(unsigned int* const d_inputVals,
           unsigned int* const d_inputPos,
           unsigned int* const d_outputVals,
           unsigned int* const d_outputPos,
           const size_t numElems)
{
    unsigned int *d_inVals;
    unsigned int *d_inPos;
    unsigned int *d_zerosPredicate;
    unsigned int *d_onesPredicate;
    unsigned int *d_zerosScan;
    unsigned int *d_onesScan;
    size_t memsize = sizeof(unsigned int) * numElems;
    size_t numBlocks = CEILING_DIVIDE(numElems, BLOCK_WIDTH);

    cudaMalloc(&d_inVals, memsize);
    cudaMalloc(&d_inPos, memsize);
    cudaMalloc(&d_zerosPredicate, memsize);
    cudaMalloc(&d_onesPredicate, memsize);
    cudaMalloc(&d_zerosScan, memsize);
    cudaMalloc(&d_onesScan, memsize);

    cudaMemcpy(d_inVals, d_inputVals, memsize, cudaMemcpyDeviceToDevice);
    cudaMemcpy(d_inPos, d_inputPos, memsize, cudaMemcpyDeviceToDevice);

    for(unsigned int bit = 0; bit < 32; bit++) {
        cudaMemset(d_zerosScan, 0, memsize);
        cudaMemset(d_onesScan, 0, memsize);

        mapPredicate<<<numBlocks, BLOCK_WIDTH>>>(
            d_zerosPredicate,
            d_onesPredicate,
            d_inVals,
            bit,
            numElems
        );
        
        totalScan(d_zerosPredicate, d_zerosScan, numElems);
        totalScan(d_onesPredicate, d_onesScan, numElems);

        scatter<<<numBlocks, BLOCK_WIDTH>>>(
            d_inVals,
            d_outputVals,
            d_inPos,
            d_outputPos,
            d_zerosScan,
            d_onesScan,
            d_zerosPredicate,
            d_onesPredicate,
            numElems
        );
        cudaMemcpy(d_inVals, d_outputVals, memsize, cudaMemcpyDeviceToDevice);
        cudaMemcpy(d_inPos, d_outputPos, memsize, cudaMemcpyDeviceToDevice);
    }

    cudaFree(d_inVals);
    cudaFree(d_inPos);
    cudaFree(d_zerosPredicate);
    cudaFree(d_onesPredicate);
    cudaFree(d_zerosScan);
    cudaFree(d_onesScan);
}

int main(int argc, char **argv) {

  wbArg_t args;
   args = wbArg_read(argc, argv);
    float *h_inVals;
    float *h_inPos;
    float *h_outVals;
    float *h_outPos;
    size_t memsize = sizeof(unsigned int) * TEST_SIZE;
    
     int numElements; 
    
  
    
    wbTime_start(Generic, "Importing data and creating memory on host");
    h_inVals = (float *) wbImport(wbArg_getInputFile(args, 0), &numElements);
   

   
    h_inPos = (float *)malloc(memsize);
    h_outVals = (float *)malloc(memsize);
    h_outPos = (float *)malloc(memsize);
    
  
    for(int i=0; i<TEST_SIZE; i++){ 
    	h_inPos[i] = i; 
    }
 //wbTime_stop(Generic, "Importing data and creating memory on host");

 
   printf("\nInput Length :%d\n",numElements);

    unsigned int *d_inputVals;
    unsigned int *d_inputPos;
    unsigned int *d_outputVals;
    unsigned int *d_outputPos;
    
   // wbTime_start(GPU, "Allocating GPU memory.");
    cudaMalloc(&d_inputVals, memsize);
    cudaMalloc(&d_inputPos, memsize);
    cudaMalloc(&d_outputVals, memsize);
    cudaMalloc(&d_outputPos, memsize);
    
   // wbTime_stop(GPU, "Allocating GPU memory.");
    
   //  wbTime_start(GPU, "Copying input memory to the GPU.");
    cudaMemcpy(d_inputVals, h_inVals, memsize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_inputPos, h_inPos, memsize, cudaMemcpyHostToDevice);
  //  wbTime_stop(GPU, "Copying input memory to the GPU.");
    
  //  wbTime_start(Compute, "Performing CUDA computation");
	int start_s=clock();
	radix(d_inputVals, d_inputPos, d_outputVals, d_outputPos, TEST_SIZE);
	int stop_s=clock();
	
	// wbTime_stop(Compute, "Performing CUDA computation");

  //    wbTime_start(Copy, "Copying output memory to the CPU");
 	cudaMemcpy(h_outVals, d_outputVals, memsize, cudaMemcpyDeviceToHost);
    	cudaMemcpy(h_outPos, d_outputPos, memsize, cudaMemcpyDeviceToHost);
    //	wbTime_stop(Copy, "Copying output memory to the CPU");
  //  wbTime_start(GPU, "Freeing GPU Memory");
    cudaFree(d_inputVals);
    cudaFree(d_inputPos);
    cudaFree(d_outputVals);
    cudaFree(d_outputPos);
  //  wbTime_stop(GPU, "Freeing GPU Memory");
  printf("\n");
      wbSolution(args, h_outVals, numElements);
      printf("\nTime :  %f s \n",(stop_s-start_s)/double(CLOCKS_PER_SEC));
printf("\n");
    free(h_inVals);
    free(h_inPos);
    free(h_outVals);
    free(h_outPos);
    return 0;
}

