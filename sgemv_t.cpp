#include <assert.h>
#include <stdio.h>
#include <algorithm>
#include <stdlib.h>
#include<iostream>
#include "hip/hip_runtime.h"


#define HIP_ASSERT(x) (assert((x)==hipSuccess))


#define M    (8192)
#define N    (2048)
#define NN   (8192)

#define NUM       (M*NN)

#define THREADS_PER_BLOCK_X  16
#define THREADS_PER_BLOCK_Y  32
#define THREADS_PER_BLOCK_Z  1


//Marix  A in normal format:  M*N
//Matrix B in N *1  or 1 *N
//Matrix C will be M * 1
//WorkGroup_SIZE  16 * 32
//WK Group TIlE： result C in 16 * 1 
//WK Grouop       Matrix A in 16 * N,   Matrix B in N * 1 /
// Thread Tile  per iteration : 1 * 4 * 1
// each thread read 4x rows.
// Wave per Each loop:  16 x (1x(4x4)x1)
// Work Group per Loop  16 x (1x(8x4x4)x1) 

//Mehtod 2 : Load B into Shared Memory 
//One wave needs 16 DWORDs per loop

__global__ void sgemv_t_c16x1_t1x4x1(const float* a, const float* b, float* __restrict__ c, const int m, const int n, const int lda ){
#define X_FMA 4    
    int tile_m  =  hipBlockIdx_x  * 16 + hipThreadIdx_x;   
    int row_off = hipThreadIdx_y  * X_FMA;  

    if(tile_m >= m)
        return;

    const float* a_ptr = a + tile_m + row_off * lda;
    const float* b_ptr = b + row_off;    

    float sum= 0;
    int i =0;
    float adata[X_FMA];
    float bdata[X_FMA];

    float shared_bdata[64];
    //4X data alignment per thread
    //32x data alignment per workgroup

    for(i=0; (i+row_off+X_FMA) < n; i+= (32 * X_FMA) )    
    {
        for(int j=0;j < X_FMA; j++){
            bdata[j] = b_ptr[i+j];
            //asm volatile ("s_nop 1\n");
        }
        //read A/B 4x rows 
        for(int j=0;j < X_FMA; j++){
            adata[j] = a_ptr[(i+j) * lda];            
         }
         
        //read B 4x data 
        for(int j=0;j < X_FMA; j++){
            sum += adata[j] * bdata[j];
        }
    }
    //Last 3 
    {
        adata[0] = 0;
        bdata[0] = 0;
        if((i + row_off) < n){
            adata[0] = a_ptr[(i + 0)* lda];    
            bdata[0] = b_ptr[ i + 0];
        }
        adata[1] = 0;
        bdata[1] = 0;
        if((i +1 + row_off) < n){
            adata[1] = a_ptr[(i + 1)* lda];    
            bdata[1] = b_ptr[ i + 1];
        }
        adata[2] = 0;
        bdata[2] = 0;
        if((i + 2 + row_off) < n){
            adata[2] = a_ptr[(i + 2)* lda];    
            bdata[2] = b_ptr[ i + 2];
        }

        sum += adata[0] * bdata[0];            
        sum += adata[1] * bdata[1];            
        sum += adata[2] * bdata[2];           
    }

    //reduction
    __shared__ float sum_shared[ 16 * 32 ];

    int idx = hipThreadIdx_x +  hipThreadIdx_y * 16;
    sum_shared[ idx ]  = sum;
    __syncthreads();
    //reduction to 64 threads
    
    if(idx < 64)
    {
        sum = sum_shared[idx + 64 * 0];
        sum += sum_shared[idx + 64 * 1];
        sum += sum_shared[idx + 64 * 2];
        sum += sum_shared[idx + 64 * 3];
        sum += sum_shared[idx + 64 * 4];
        sum += sum_shared[idx + 64 * 5];
        sum += sum_shared[idx + 64 * 6];
        sum += sum_shared[idx + 64 * 7];
        sum_shared[idx  ]  = sum;
    }

    if(idx < 16)
    {
        sum = sum_shared[idx + 16 * 0];
        sum += sum_shared[idx + 16 * 1];
        sum += sum_shared[idx + 16 * 2];
        sum += sum_shared[idx + 16 * 3];
        
        c[tile_m]  = sum;        
    }
}


__global__ void sgemv_t_c64x1_t1x2x1(const float* a, const float* b, float* __restrict__ c, const int m, const int n, const int lda ){
#undef X_FMA
#define X_FMA 2    
    int tile_m  =  hipBlockIdx_x  * 64 + hipThreadIdx_x;   
    int row_off = hipThreadIdx_y * X_FMA;  

    if(tile_m >= m)
        return;

    const float* a_ptr = a + tile_m + row_off * lda;
    const float* b_ptr = b + row_off;    

    float sum = 0;
    int i =0;
    float adata[X_FMA];
    float bdata[X_FMA];

    float shared_bdata[64];
    //4X data alignment per thread
    //32x data alignment per workgroup

    for(i=0; (i+row_off+X_FMA) < n; i+= ( 8  * X_FMA) )    
    {
        //read A/B 4x rows 
        for(int j=0;j < X_FMA; j++){
            adata[j] = a_ptr[(i+j) * lda ];            
         }
         
        //read B 4x data 
        for(int j=0;j < X_FMA; j++){
            bdata[j] = b_ptr[i+j];
        }
        for(int j=0;j < X_FMA; j++){
            sum += adata[j]   * bdata[j];
        }
    }
    //Last 1 
    {
        adata[0] = 0;
        bdata[0] = 0;
        if((i + row_off) < n){
            adata[0] = a_ptr[(i + 0)* lda];    
            bdata[0] = b_ptr[ i + 0];
        }
        sum += adata[0] * bdata[0];            
    }

    //reduction
    __shared__ float sum_shared[ 8 * 64  ];

    int idx = hipThreadIdx_x +  hipThreadIdx_y * 64;
    sum_shared[ idx ]  = sum;
    __syncthreads();
    //reduction to 64 threads
    
    if(idx < 64)
    {
        sum = sum_shared[idx + 64 * 0];
        sum += sum_shared[idx + 64 * 1];
        sum += sum_shared[idx + 64 * 2];
        sum += sum_shared[idx + 64 * 3];
        sum += sum_shared[idx + 64 * 4];
        sum += sum_shared[idx + 64 * 5];
        sum += sum_shared[idx + 64 * 6];
        sum += sum_shared[idx + 64 * 7];
        c[tile_m+0]  = sum;        
    }    
}



using namespace std;
int main() {
  
  float* hostA;
  float* hostB;
  float* hostC;

  float* deviceA;
  float* deviceB;
  float* deviceC;

  hipDeviceProp_t devProp;
  hipGetDeviceProperties(&devProp, 0);
  cout << " System minor " << devProp.minor << endl;
  cout << " System major " << devProp.major << endl;
  cout << " agent prop name " << devProp.name << endl;



  cout << "hip Device prop succeeded " << endl ;


  int i;
  int errors;

  hostA = (float*)malloc(NUM * sizeof(float));
  hostB = (float*)malloc(N * sizeof(float));
  hostC = (float*)malloc(M * sizeof(float));
  
  // initialize the input data
  for (i = 0; i < NUM; i++) {
    hostA[i] = (float)sinf(i);
  }
  for (i = 0; i < N; i++) {
    hostB[i] = (float)cosf(i);
  }

  
  HIP_ASSERT(hipMalloc((void**)&deviceA, NUM * sizeof(float)));
  HIP_ASSERT(hipMalloc((void**)&deviceB, N * sizeof(float)));
  HIP_ASSERT(hipMalloc((void**)&deviceC, M * sizeof(float)));
  
  HIP_ASSERT(hipMemcpy(deviceA, hostA, NUM*sizeof(float), hipMemcpyHostToDevice));
  HIP_ASSERT(hipMemcpy(deviceB, hostB, N*sizeof(float), hipMemcpyHostToDevice));

  hipEvent_t start, stop;

	hipEventCreate(&start);
	hipEventCreate(&stop);
	float eventMs = 0.0f;
  
   
   for(int mm=512; mm <=M; mm+=256)
   {
          int n_off = ((NN / 4) + 31)  & 0xFFFFFFE0;
          hipLaunchKernelGGL(sgemv_t_c16x1_t1x4x1, 
                        dim3((mm/16)),
                        dim3(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y),
                        0, 0,
                        deviceA ,deviceB ,deviceC, mm, NN, mm);

          hipEventRecord(start, NULL);
        for (int i = 0; i < 1; i++){
          hipLaunchKernelGGL(sgemv_t_c16x1_t1x4x1, 
                        dim3((mm/16) ),
                        dim3(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y),
                        0, 0,
                        deviceA ,deviceB ,deviceC, mm, NN, mm);
        }

          hipEventRecord(stop, NULL);
          hipEventSynchronize(stop);

          hipEventElapsedTime(&eventMs, start, stop);
   
		      //printf("elapsed time:%f\n", eventMs);
          double total_bytes = ( double)(mm)* (double)NN + double(mm) + double(NN);          
          total_bytes = total_bytes * sizeof(float) /1024/1024/1024;
		  double gbps = total_bytes/eventMs * 1000 * 1;
		  printf("sgemv_t_c16x1_t1x4x1 [mm=%d] ==> %lf Gi Bytes/s, ms: %f\n", mm, gbps, eventMs);
   }
   for(int mm=512; mm <=M; mm+=256)
   {
          hipLaunchKernelGGL(sgemv_t_c64x1_t1x2x1, 
                        dim3((mm/64)),
                        dim3(64, 8),
                        0, 0,
                        deviceA ,deviceB ,deviceC, mm, NN, mm);

          hipEventRecord(start, NULL);
        for (int i = 0; i < 1; i++){
          hipLaunchKernelGGL(sgemv_t_c64x1_t1x2x1, 
                        dim3((mm/64) ),
                        dim3(64,8),
                        0, 0,
                        deviceA ,deviceB ,deviceC, mm, NN, mm);
        }

          hipEventRecord(stop, NULL);
          hipEventSynchronize(stop);

          hipEventElapsedTime(&eventMs, start, stop);
   
		      //printf("elapsed time:%f\n", eventMs);
          double total_bytes = ( double)(mm)* (double)NN + double(mm) + double(NN);          
          total_bytes = total_bytes * sizeof(float) /1024/1024/1024;
		  double gbps = total_bytes/eventMs * 1000 * 1;
		  printf("sgemv_t_c64x1_t1x2x1 [mm=%d] ==> %lf Gi Bytes/s, ms: %f\n", mm, gbps, eventMs);
   }

  HIP_ASSERT(hipMemcpy(hostC, deviceC, M*sizeof(float), hipMemcpyDeviceToHost));

  // verify the results

  HIP_ASSERT(hipFree(deviceA));
  HIP_ASSERT(hipFree(deviceB));
  HIP_ASSERT(hipFree(deviceC));

  free(hostA);
  free(hostB);
  free(hostC);

  //hipResetDefaultAccelerator();

  return errors;
}
