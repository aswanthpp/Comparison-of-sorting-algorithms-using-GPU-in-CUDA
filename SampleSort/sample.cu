/* This program sorts an input array by bucket sort.
 * Each bucket in turn is sorted using Parallel Bubble sort.
 * The array consists of float numbers, all less than 1. To find the destination bucket,
 * the float number is multiplied by 10 to get the first digit, which determines the bucket number.
 * For eg., 0.1234 -> (int)(0.1234*10) = 1. Thus the bucket number for 0.1234 is 1.
 * Thus the total number of buckets will be 10. (0-9)
 * Implemented in CUDA. 
 *
 *
 * 
 * code by Anand Goyal. Dated: 12/13/2014
*/

#include<stdio.h>
#include<cuda.h>
#include<time.h>
#include<sys/time.h>
#include"wb.h"
#define range 10
#define SIZE 50000
#define bucketLength (SIZE/range * 2)

__global__ void bucketSortKernel(float *inData, long size, float *outData)
{
	__shared__ float localBucket[bucketLength];
	__shared__ int localCount; /* Counter to track index with a bucket */

	int tid = threadIdx.x; int blockId = blockIdx.x;
	int offset = blockDim.x;
	int bucket, index, phase;
	float temp;
	
	if(tid == 0)
		localCount = 0;

	__syncthreads();

	/* Block traverses through the array and buckets the element accordingly */
	while(tid < size) {
		bucket = inData[tid] * 10;
		if(bucket == blockId) {
			index = atomicAdd(&localCount, 1);
			localBucket[index] = inData[tid]; 
		}
		tid += offset;		
	}

	__syncthreads();
	
	tid = threadIdx.x;
	//Sorting the bucket using Parallel Bubble Sort
	for(phase = 0; phase < bucketLength; phase ++) {
		if(phase % 2 == 0) {
			while((tid < bucketLength) && (tid % 2 == 0)) {
				if(localBucket[tid] > localBucket[tid +1]) {
					temp = localBucket[tid];
					localBucket[tid] = localBucket[tid + 1];
					localBucket[tid + 1] = temp;
				}
				tid += offset;
			}
		}
		else {
			while((tid < bucketLength - 1) && (tid %2 != 0)) {
				if(localBucket[tid] > localBucket[tid + 1]) {
					temp = localBucket[tid];
					localBucket[tid] = localBucket[tid + 1];
					localBucket[tid + 1] = temp;
				}
				tid += offset;
			}
		}
	}
	
	tid = threadIdx.x;
	while(tid < bucketLength) {
		outData[(blockIdx.x * bucketLength) + tid] = localBucket[tid];
		tid += offset;
	}
}

int main(int argc, char **argv) {

  wbArg_t args;
   args = wbArg_read(argc, argv);
	float *input, *output;
	float *d_input, *d_output;
	float elapsedTime;
	cudaEvent_t start, stop;
	
	cudaEventCreate(&start, 0);
	cudaEventCreate(&stop, 0);

	/* Each block sorts one bucket */
	const int numOfThreads = 4;
	const int numOfBlocks = range;

	input = (float *)malloc(sizeof(float) * SIZE);
	output = (float *)malloc(sizeof(float) * bucketLength * range);
	cudaMalloc((void**)&d_input, sizeof(float) * SIZE);
	cudaMalloc((void **)&d_output, sizeof(float) * bucketLength * range);
	cudaMemset(d_output, 0, sizeof(float) * bucketLength * range);
	
	
	 int numElements; 
    
  
    
    
    	input = (float *) wbImport(wbArg_getInputFile(args, 0), &numElements);
   	printf("\nInput Length : %d\n",numElements);

	cudaEventRecord(start, 0);

	cudaMemcpy(d_input, input, sizeof(float) * SIZE, cudaMemcpyHostToDevice);
	bucketSortKernel<<<numOfBlocks, numOfThreads>>>(d_input, SIZE, d_output);
	cudaMemcpy(output, d_output, sizeof(float) * bucketLength * range, cudaMemcpyDeviceToHost);
	
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&elapsedTime, start, stop);
	
	/*int flag=0;
	for(int i=0;i<numElements-1;i++){
		if(output[i]>output[i+1]){ 
		//and (output[i] !=0 or output[i+1]!=0)){
			printf("\nSolution is incorrect \n");
			flag=1;
			//printf("\n %d value %f next %f ---\n",i,output[i],output[i+1]);
			break;
		}
	}
	if(flag==0){
		printf("\nSolution is Correct !!!\n");
	
 	}
 	*/	
 	printf("\nTime :  %3.1f ms \n", elapsedTime);
	cudaFree(d_input);
	cudaFree(d_output);
	free(input);
	free(output);

	return 0;
}

