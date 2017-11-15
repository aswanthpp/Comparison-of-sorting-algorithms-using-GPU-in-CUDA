#include <iostream>
#include <stdlib.h>
#include "helper_cuda.h"
#include <sys/time.h>
#include "wb.h"
using namespace std;

long readList(long**);

void mergesort(float*, dim3, dim3);
__global__ void gpu_mergesort(float*, float*, long, long, dim3*, dim3*);
__device__ void gpu_bottomUpMerge(float*, float*, long, long, long);

#define min(a, b) (a < b ? a : b)
#define size 10000

bool verbose;
int main(int argc, char** argv) 
{
    wbArg_t args;
    args = wbArg_read(argc, argv);
    clock_t start,end;
    double cput;

    start = clock();

    dim3 threadsPerBlock;
    dim3 blocksPerGrid;

    threadsPerBlock.x = 32;
    threadsPerBlock.y = 1;
    threadsPerBlock.z = 1;

    blocksPerGrid.x = 8;
    blocksPerGrid.y = 1;
    blocksPerGrid.z = 1;

    if (verbose) {
        cout << "\nthreadsPerBlock:"
                  << "\n  x: " << threadsPerBlock.x
                  << "\n  y: " << threadsPerBlock.y
                  << "\n  z: " << threadsPerBlock.z
                  << "\n\nblocksPerGrid:"
                  << "\n  x:" << blocksPerGrid.x
                  << "\n  y:" << blocksPerGrid.y
                  << "\n  z:" << blocksPerGrid.z
                  << "\n\n total threads: " 
                  << threadsPerBlock.x * threadsPerBlock.y * threadsPerBlock.z *
                     blocksPerGrid.x * blocksPerGrid.y * blocksPerGrid.z
                  << "\n\n";
                  
               
    }
    int inputLength;
     float  *data;
    data = (float *)wbImport(wbArg_getInputFile(args, 0), &inputLength);

    mergesort(data, threadsPerBlock, blocksPerGrid);
   
    wbSolution(args, data, inputLength);

    cout<<"\nInput Length : "<<size<<endl;
    
    end = clock();
    cput = ((double)(end-start))/CLOCKS_PER_SEC;
    cout<<"\nRunning time = " << cput <<" s"<< endl;
}

void mergesort(float * data, dim3 threadsPerBlock, dim3 blocksPerGrid) 
{
    float* D_data;
    float* D_swp;
    dim3* D_threads;
    dim3* D_blocks;
    
    checkCudaErrors(cudaMalloc((void**) &D_data, size * sizeof(long)));
    checkCudaErrors(cudaMalloc((void**) &D_swp, size * sizeof(long)));

    checkCudaErrors(cudaMemcpy(D_data, data, size * sizeof(long), cudaMemcpyHostToDevice));
 
    checkCudaErrors(cudaMalloc((void**) &D_threads, sizeof(dim3)));
    checkCudaErrors(cudaMalloc((void**) &D_blocks, sizeof(dim3)));

    checkCudaErrors(cudaMemcpy(D_threads, &threadsPerBlock, sizeof(dim3), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(D_blocks, &blocksPerGrid, sizeof(dim3), cudaMemcpyHostToDevice));

    float* A = D_data;
    float* B = D_swp;

    long nThreads = threadsPerBlock.x * threadsPerBlock.y * threadsPerBlock.z *
                    blocksPerGrid.x * blocksPerGrid.y * blocksPerGrid.z;

    for (int width = 2; width < (size << 1); width <<= 1) {
        long slices = size / ((nThreads) * width) + 1;

        if (verbose) {
            cout << "mergeSort - width: " << width 
                      << ", slices: " << slices 
                      << ", nThreads: " << nThreads << '\n';
        }

        gpu_mergesort<<<blocksPerGrid, threadsPerBlock>>>(A, B, width, slices, D_threads, D_blocks);

        A = A == D_data ? D_swp : D_data;
        B = B == D_data ? D_swp : D_data;
    }

    checkCudaErrors(cudaMemcpy(data, A, size * sizeof(long), cudaMemcpyDeviceToHost));
    
        checkCudaErrors(cudaFree(A));
    checkCudaErrors(cudaFree(B));
}

__device__ unsigned int getIdx(dim3* threads, dim3* blocks) 
{
    int x;
    return threadIdx.x +
           threadIdx.y * (x  = threads->x) +
           threadIdx.z * (x *= threads->y) +
           blockIdx.x  * (x *= threads->z) +
           blockIdx.y  * (x *= blocks->z) +
           blockIdx.z  * (x *= blocks->y);
}


__global__ void gpu_mergesort(float* source, float* dest, long width, long slices, dim3* threads, dim3* blocks) 
{
    unsigned int idx = getIdx(threads, blocks);
    long start = width*idx*slices, 
         middle, 
         end;

    for (long slice = 0; slice < slices; slice++) 
    {
        if (start >= size)
            break;

        middle = min(start + (width >> 1), size);
        end = min(start + width, size);
        gpu_bottomUpMerge(source, dest, start, middle, end);
        start += width;
    }
}

__device__ void gpu_bottomUpMerge(float* source, float* dest, long start, long middle, long end) 
{
    long i = start;
    long j = middle;
    for (long k = start; k < end; k++) {
        if (i < middle && (j >= end || source[i] < source[j])) 
        {
            dest[k] = source[i];
            i++;
        } 
        else 
        {
            dest[k] = source[j];
            j++;
        }
    }
}
