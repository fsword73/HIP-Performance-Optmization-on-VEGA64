#include <assert.h>
#include <stdio.h>
#include <algorithm>
#include <stdlib.h>
#include <unistd.h>
#include<iostream>
#include "hip/hip_runtime.h"


#define HIP_ASSERT(x) (assert((x)==hipSuccess))


#define M    8192
#define N    8192
#define K    (8192)

#define NUM       (M*K)

#define THREADS_PER_BLOCK_X  16
#define THREADS_PER_BLOCK_Y  16
#define THREADS_PER_BLOCK_Z  1

__global__ void sgemm_nt_128x128(const float* a, const float* b, float* __restrict__ c, const int m, const int n, const int k ){
  int wk_tile_m =  hipBlockIdx_y * 128 ;
  int wk_tile_n =  hipBlockIdx_x * 128 ;
  int thread_tile_m = wk_tile_m + hipThreadIdx_y * 8;
  int thread_tile_n = wk_tile_n + hipThreadIdx_x * 8;

  float sum[8][8];

  for(int i=0; i < 8; i++){
      for(int j=0; j < 8; j++){
        sum[i][j] = 0;
      }
  }

  for(int kk=0; kk < k; kk++) {
      float adata[8];
      float bdata[8];

      for(int i=0; i < 8; i++) {
         adata[i] = a[( thread_tile_m + i)* k +kk];
      }
      for(int i=0; i < 8; i++) {
         bdata[i] = b[( thread_tile_n + i) *k +kk];
      }     

      for(int i=0; i <8; i++){
          for(int j=0; j <8; j++){
             sum[i][j] += adata[i] * bdata[j]; 
          }
      }

  } 

  //store   
  for(int i=0; i < 8; i++){
      for(int j=0; j < 8; j++){
          c[ (thread_tile_m + i) * n + thread_tile_n + j] = sum[i][j];
      }
  }  
}


__global__ void sgemm_nt_128x128_unroll2(const float* a, const float* b, float* __restrict__ c, const int m, const int n, const int k ){
  int wk_tile_m =  hipBlockIdx_y * 128 ;
  int wk_tile_n =  hipBlockIdx_x * 128 ;
  int thread_tile_m = wk_tile_m + hipThreadIdx_y * 8;
  int thread_tile_n = wk_tile_n + hipThreadIdx_x * 8;

  float sum[8][8];

  for(int i=0; i < 8; i++){
      for(int j=0; j < 8; j++){
        sum[i][j] = 0;
      }
  }

  for(int kk=0; kk < k; kk+=2) {
      float adata[8*2];
      float bdata[8*2];

      for(int i=0; i < 8; i++) {
         adata[i] = a[( thread_tile_m + i)* k +kk];
         adata[i+8] = a[( thread_tile_m + i)* k +kk+1];
      }
      for(int i=0; i < 8; i++) {
         bdata[i] = b[( thread_tile_n + i) *k +kk];
         bdata[i+8] = b[( thread_tile_n + i) *k +kk+1];
      }     

      for(int i=0; i <8; i++){
          for(int j=0; j <8; j++){
             sum[i][j] += adata[i] * bdata[j]; 
             sum[i][j] += adata[i+8] * bdata[j+8]; 
          }
      }
  } 

  //store   
  for(int i=0; i < 8; i++){
      for(int j=0; j < 8; j++){
          c[ (thread_tile_m + i) * n + thread_tile_n + j] = sum[i][j];
      }
  }  
}





__global__ void sgemm_nt_128x128_lds_unroll8(const float* a, const float* b, float* __restrict__ c, const int m, const int n, const int k ){
  int wk_tile_m =  hipBlockIdx_y * 128 ;
  int wk_tile_n =  hipBlockIdx_x * 128 ;
  int thread_tile_m = wk_tile_m + hipThreadIdx_y * 8;
  int thread_tile_n = wk_tile_n + hipThreadIdx_x * 8;
  int local_id = hipThreadIdx_y * 16 + hipThreadIdx_x;
  int local_tile_m, local_tile_n;
  local_tile_m = hipThreadIdx_x * 8;
  local_tile_n = hipThreadIdx_y * 8;

  float sum[8][8];
  __shared__ float a_shared[128*8];
  __shared__ float b_shared[128*8];

  for(int i=0; i < 8; i++){
      for(int j=0; j < 8; j++){
        sum[i][j] = 0;
      }
  }
 
  float* ptr = NULL;
  int local_write = 0;
  //first 128 threads load A, next 128 thread load B
  if(local_id < 128)
  {
    ptr = (float*)(a + (wk_tile_m + local_id) * k); 
    local_write = local_id;
  }
  else 
  {
     ptr = (float*)(b + (wk_tile_n + local_id-128) * k); 
     local_write = local_id -128;
  }


  //unroll 8
  for(int kk=0; kk < k; kk+=8) {

      //stroed into LDS
      if(local_id < 128)
      {
          for(int i=0; i < 8; i++)
          {
            a_shared[i*128 + local_write] = ptr[i+kk];
          }
      }
      else 
      {
          for(int i=0; i < 8; i++)
          {
            b_shared[i*128 + local_write] = ptr[i+kk];
          }
      }
      __syncthreads();      

      //8x8x8 FMAs
      for(int s=0; s < 8; s++)  
      {     float adata[8];
            float bdata[8];

            for(int t=0; t < 8; t++){
              adata[t] = a_shared[local_tile_m + t + s * 128];
              bdata[t] = b_shared[local_tile_n + t + s * 128];
            }
            for(int i=0; i <8; i++){          
                for(int j=0; j <8; j++){
                  sum[i][j] += adata[i] * bdata[j]; 
                }
            }
      }      
  } 

  //store   
  for(int i=0; i < 8; i++){
      for(int j=0; j < 8; j++){
          c[ (thread_tile_m + i) * n + thread_tile_n + j] = sum[i][j];
      }
  }  
}

__global__ void sgemm_nt_128x128_lds_unroll8_double_buf(const float* a, const float* b, float* __restrict__ c, const int m, const int n, const int k ){
  int wk_tile_m =  hipBlockIdx_y * 128 ;
  int wk_tile_n =  hipBlockIdx_x * 128 ;
  int thread_tile_m = wk_tile_m + hipThreadIdx_y * 8;
  int thread_tile_n = wk_tile_n + hipThreadIdx_x * 8;
  int local_id = hipThreadIdx_y * 16 + hipThreadIdx_x;
  int local_tile_m, local_tile_n;
  local_tile_m = hipThreadIdx_x * 8;
  local_tile_n = hipThreadIdx_y * 8;

  float sum[8][8];
  __shared__ float a_shared[128*8];
  __shared__ float b_shared[128*8];

  for(int i=0; i < 8; i++){
      for(int j=0; j < 8; j++){
        sum[i][j] = 0;
      }
  }
 
  float* ptr_a = NULL;
  float* ptr_b = NULL;
  int local_write = 0;
  //First Fetch
  //128x4 data scattered into 256 threads  
  //every threads fetch DWORDX2

  //local memory store 128 data 
  if(local_id < 128 )
  {
      ptr_a = (float*)(a + (wk_tile_m + local_id ) * k);  
      ptr_b = (float*)(b + (wk_tile_n + local_id ) * k);  
      local_write = local_id;
  }
  else
  {
       //offset K + 2 == 2,3
       ptr_a = (float*)(a + (wk_tile_m + (local_id &0x7f) ) * k + 2 );  
       ptr_b = (float*)(b + (wk_tile_n + (local_id &0x7f) ) * k + 2 );  
       local_write = local_id + 128*2;
  }

  
  //fetch offset 0,1,2,3
  a_shared[local_write ]      = *(ptr_a + 0);
  a_shared[local_write +128 ] = *(ptr_a + 1);
  b_shared[local_write ]      = *(ptr_b + 0);
  b_shared[local_write +128 ] = *(ptr_b + 1);

  //unroll 8
  for(int kk=0; kk < k-8; kk+=8) {
      __syncthreads();      

      //fetch offset: kk+  4,5,6,7
      a_shared[local_write + 128 *4 ]       = *(ptr_a + kk + 4);
      a_shared[local_write + 128 *4 + 128 ] = *(ptr_a + kk + 5);
      b_shared[local_write + 128 *4]        = *(ptr_b + kk + 4);
      b_shared[local_write + 128 *4 + 128 ] = *(ptr_b + kk + 5);

      //8x8x4 FMAs
      for(int s=0; s < 4; s++)  
      {     float adata[8];
            float bdata[8];

            for(int t=0; t < 8; t++){
              adata[t] = a_shared[local_tile_m + t + s * 128];
              bdata[t] = b_shared[local_tile_n + t + s * 128];
            }
            for(int i=0; i <8; i++){          
                for(int j=0; j <8; j++){
                  sum[i][j] += adata[i] * bdata[j]; 
                }
            }
      }      

      __syncthreads();
      //fetch offset kk+ 8,9,10,11
      a_shared[local_write + 128 *4 ]       = *(ptr_a + kk + 8);
      a_shared[local_write + 128 *4 + 128 ] = *(ptr_a + kk + 9);
      b_shared[local_write + 128 *4]        = *(ptr_b + kk + 8);
      b_shared[local_write + 128 *4 + 128 ] = *(ptr_b + kk + 9);
      for(int s=4; s < 8; s++)  
      {     float adata[8];
            float bdata[8];

            for(int t=0; t < 8; t++){
              adata[t] = a_shared[local_tile_m + t + s * 128];
              bdata[t] = b_shared[local_tile_n + t + s * 128];
            }
            for(int i=0; i <8; i++){          
                for(int j=0; j <8; j++){
                  sum[i][j] += adata[i] * bdata[j]; 
                }
            }
      }      

  } 

  //last 8X 
  {
      int kk= k - 8;
      __syncthreads();      

      //fetch kk  offset+ 4,5,6,7
      a_shared[local_write + 128 *4 ]       = *(ptr_a + kk + 4);
      a_shared[local_write + 128 *4 + 128 ] = *(ptr_a + kk + 5);
      b_shared[local_write + 128 *4]        = *(ptr_b + kk + 4);
      b_shared[local_write + 128 *4 + 128 ] = *(ptr_b + kk + 5);

      //8x8x4 FMAs
      for(int s=0; s < 4; s++)  
      {     float adata[8];
            float bdata[8];

            for(int t=0; t < 8; t++){
              adata[t] = a_shared[local_tile_m + t + s * 128];
              bdata[t] = b_shared[local_tile_n + t + s * 128];
            }
            for(int i=0; i <8; i++){          
                for(int j=0; j <8; j++){
                  sum[i][j] += adata[i] * bdata[j]; 
                }
            }
      }      

      __syncthreads();
      for(int s=4; s < 8; s++)  
      {     float adata[8];
            float bdata[8];

            for(int t=0; t < 8; t++){
              adata[t] = a_shared[local_tile_m + t + s * 128];
              bdata[t] = b_shared[local_tile_n + t + s * 128];
            }
            for(int i=0; i <8; i++){          
                for(int j=0; j <8; j++){
                  sum[i][j] += adata[i] * bdata[j]; 
                }
            }
      }      
  } 

  //store   
  for(int i=0; i < 8; i++){
      for(int j=0; j < 8; j++){
          c[ (thread_tile_m + i) * n + thread_tile_n + j] = sum[i][j];
      }
  }  
}

/*
Matrix A: M * K 
Matrix B: Transposed N * K 
Matrix C: M * N  
Workgroup TILE SIZE:  96 * 128
Thread    TILE SIZE:  6 * 8    
Deal with Right , bottom 
*/
__global__ void sgemm_nt_96x128(const float* a, const float* b, float* __restrict__ c, const int m, const int n, const int k ){
  int wk_tile_m =  hipBlockIdx_y * 96 ;
  int wk_tile_n =  hipBlockIdx_x * 128 ;
  int thread_tile_m = wk_tile_m + hipThreadIdx_y * 6;
  int thread_tile_n = wk_tile_n + hipThreadIdx_x * 8;

  float sum[6][8];
#if 1 
  for(int i=0; i < 6; i++){
      for(int j=0; j < 8; j++){
        sum[i][j] = 0;      
      }
  }
#else 
  for(int i=0; i < 6; i++){
      for(int j=0; j < 8; j++){
        sum[i][j] = 0;      
      }
  }

#endif 
  for(int kk=0; kk < k; kk++) {
      float adata[6];
      float bdata[8];

      for(int i=0; i < 6; i++) {
        if( (thread_tile_m + i) < m){
           adata[i] = a[( thread_tile_m + i)* k +kk];
        }
        else
        {
          adata[i] = 0;
        }
      }
      for(int i=0; i < 8; i++) {
        if(( thread_tile_n + i) < n){
         bdata[i] = b[( thread_tile_n + i) *k +kk];
        }
        else{
          bdata[i] = 0;
        }
      }     

      for(int i=0; i <6; i++){
          for(int j=0; j <8; j++){
             sum[i][j] += adata[i] * bdata[j]; 
          }
      }
  } 

  //store   
  for(int i=0; i < 6; i++){
      for(int j=0; j < 8; j++){
          if((thread_tile_m + i) < m && (thread_tile_n+ j) < n){
              c[ (thread_tile_m + i) * n + thread_tile_n + j] = sum[i][j];
          }
      }
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
  hostB = (float*)malloc(NUM * sizeof(float));
  hostC = (float*)malloc(NUM * sizeof(float));
  
  // initialize the input data
  for (i = 0; i < NUM; i++) {
    hostA[i] = (float)sin(i);
    hostB[i] = (float)cos(i);
  }
  
  HIP_ASSERT(hipMalloc((void**)&deviceA, NUM * sizeof(float)));
  HIP_ASSERT(hipMalloc((void**)&deviceB, NUM * sizeof(float)));
  HIP_ASSERT(hipMalloc((void**)&deviceC, NUM * sizeof(float)));
  
  HIP_ASSERT(hipMemcpy(deviceA, hostA, NUM*sizeof(float), hipMemcpyHostToDevice));
  HIP_ASSERT(hipMemcpy(deviceB, hostB, NUM*sizeof(float), hipMemcpyHostToDevice));

  hipEvent_t start, stop;

	hipEventCreate(&start);
	hipEventCreate(&stop);
	float eventMs = 1.0f;

   for(int mnk=128;mnk<M+1; mnk+=128)
   {
          hipLaunchKernelGGL(sgemm_nt_128x128, 
                        dim3(mnk/128, mnk/128 ),
                        dim3(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y),
                        0, 0,
                        deviceA ,deviceB ,deviceC, mnk, mnk, mnk);

          hipEventRecord(start, NULL);
        for (int i = 0; i < 10; i++)
        {
          hipLaunchKernelGGL(sgemm_nt_128x128, 
                        dim3(mnk/128, mnk/128 ),
                        dim3(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y),
                        0, 0,
                        deviceA ,deviceB ,deviceC, mnk, mnk, mnk);
        }

          hipEventRecord(stop, NULL);
          hipEventSynchronize(stop);

          hipEventElapsedTime(&eventMs, start, stop);
   
		      //printf("elapsed time:%f\n", eventMs);
          double ips = ( double)(mnk)*( double)mnk*( double)mnk /1024/1024/1024 * 10;
		      ips = ips / ( double)eventMs * 1000 ;
		      printf("sgemm_nt_128x128 plain [mnk=%d]==> %lf G FMAs/s, ms: %f\n", mnk, ips, eventMs);
          usleep (500 *1000);

   }

   for(int mnk=128;mnk<M+1; mnk+=128)
   {
          hipLaunchKernelGGL(sgemm_nt_128x128_unroll2, 
                        dim3(mnk/128, mnk/128 ),
                        dim3(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y),
                        0, 0,
                        deviceA ,deviceB ,deviceC, mnk, mnk, mnk);

          hipEventRecord(start, NULL);
        for (int i = 0; i < 10; i++)
        {
          hipLaunchKernelGGL(sgemm_nt_128x128_unroll2, 
                        dim3(mnk/128, mnk/128 ),
                        dim3(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y),
                        0, 0,
                        deviceA ,deviceB ,deviceC, mnk, mnk, mnk);
        }

          hipEventRecord(stop, NULL);
          hipEventSynchronize(stop);

          hipEventElapsedTime(&eventMs, start, stop);
   
		      //printf("elapsed time:%f\n", eventMs);
          double ips = ( double)(mnk)*( double)mnk*( double)mnk /1024/1024/1024 * 10;
		      ips = ips / ( double)eventMs * 1000 ;
		      printf("sgemm_nt_128x128 unroll2 [mnk=%d] ==> %lf G FMAs/s, ms: %f\n",mnk, ips, eventMs);
          usleep (500 *1000);

   }

   for(int mnk=128;mnk<M+1; mnk+=128)
   {
          hipLaunchKernelGGL(sgemm_nt_128x128_lds_unroll8, 
                        dim3(mnk/128, mnk/128 ),
                        dim3(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y),
                        0, 0,
                        deviceA ,deviceB ,deviceC, mnk, mnk, mnk);

          hipEventRecord(start, NULL);
        for (int i = 0; i < 10; i++)
        {
          hipLaunchKernelGGL(sgemm_nt_128x128_lds_unroll8, 
                        dim3(mnk/128, mnk/128 ),
                        dim3(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y),
                        0, 0,
                        deviceA ,deviceB ,deviceC, mnk, mnk, mnk);
        }

          hipEventRecord(stop, NULL);
          hipEventSynchronize(stop);

          hipEventElapsedTime(&eventMs, start, stop);
   
		      //printf("elapsed time:%f\n", eventMs);
          double ips = ( double)(mnk)*( double)mnk*( double)mnk /1024/1024/1024*10;
		      ips = ips / ( double)eventMs * 1000 ;
		      printf("sgemm_nt_128x128_lds_unroll8:[%d x %d % d ] ==> %lf G FMAs/s, ms: %f\n", mnk,mnk,mnk, ips, eventMs);
          usleep (500 *1000);

   }

    exit(0);
   for(int mnk=128;mnk<M+1; mnk+=128)
   {
          hipLaunchKernelGGL(sgemm_nt_128x128_lds_unroll8_double_buf, 
                        dim3(mnk/128, mnk/128 ),
                        dim3(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y),
                        0, 0,
                        deviceA ,deviceB ,deviceC, mnk, mnk, mnk);

          hipEventRecord(start, NULL);
        for (int i = 0; i < 1; i++)
        {
          hipLaunchKernelGGL(sgemm_nt_128x128_lds_unroll8_double_buf, 
                        dim3(mnk/128, mnk/128 ),
                        dim3(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y),
                        0, 0,
                        deviceA ,deviceB ,deviceC, mnk, mnk, mnk);
        }

          hipEventRecord(stop, NULL);
          hipEventSynchronize(stop);

          hipEventElapsedTime(&eventMs, start, stop);
   
		      //printf("elapsed time:%f\n", eventMs);
          double ips = ( double)(mnk)*( double)mnk*( double)mnk /1024/1024/1024;
		      ips = ips / ( double)eventMs * 1000 ;
		      printf("sgemm_nt_128x128_lds_unroll8_double_buf:[%d x %d % d ] ==> %lf G FMAs/s, ms: %f\n", mnk,mnk,mnk, ips, eventMs);

   }

   if(1)
   {
          hipLaunchKernelGGL(sgemm_nt_96x128,
                        dim3((M+96-1)/96, (N+128-1)/128 ),
                        dim3(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y),
                        0, 0,
                        deviceA ,deviceB ,deviceC, M, N, K);

          hipEventRecord(start, NULL);
        for (int i = 0; i < 1; i++)
        {
          hipLaunchKernelGGL(sgemm_nt_96x128,
                        dim3((M+96-1)/96, (N+128-1)/128 ),
                        dim3(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y),
                        0, 0,
                        deviceA ,deviceB ,deviceC, M, N, K);
        }

          hipEventRecord(stop, NULL);
          hipEventSynchronize(stop);

          hipEventElapsedTime(&eventMs, start, stop);
   
		      //printf("elapsed time:%f\n", eventMs);
          double ips = ( double)(M)*( double)N*( double)K /1024/1024/1024;
		      ips = ips / ( double)eventMs * 1000 ;
		      printf("sgemm_nt_96x128 ==> %lf G FMAs/s, ms: %f\n", ips, eventMs);

   }

  HIP_ASSERT(hipMemcpy(hostC, deviceC, NUM*sizeof(float), hipMemcpyDeviceToHost));

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
