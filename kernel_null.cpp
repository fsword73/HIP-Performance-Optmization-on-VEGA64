  
//hipcc --amdgpu-target=gfx900 kernel_null.cpp  -o kernel_null
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

typedef float Float4 __attribute__((ext_vector_type(4)));

/*
inline __device__ void Matrix8x1X8(float* a,  float* b, float* c){  

      for(int i = 0; i < 8; i++)
      for(int j = 0; j < 8; j++)
      {
          asm volatile("\n \
          v_fma_f32 %0, %1, %2, %0  \n \
          "
          :
          :"v"(c[i*8+j]), "v"(a[i]), "v"(b[j])
          );
      }
}
*/


#define UNROLL_SIZE 8
__global__ void sgemm_null(const float* a, const float* b, float* __restrict__ c, const int m, const int n, const int k, const int lda, const int ldb, const double alpha, double beta ){
  int wk_tile_m =  hipBlockIdx_y * 128 ;
  int wk_tile_n =  hipBlockIdx_x * 128 ;
  int local_id = hipThreadIdx_y * 16 + hipThreadIdx_x;
  
  __shared__ float a_shared[128*UNROLL_SIZE];
  __shared__ float b_shared[128*UNROLL_SIZE];

    float sum[8*8]; 
    a_shared[local_id] = (local_id >> 6) * 1.0f;
    __syncthreads();
    for(int i=0; i <=64; i++){
        sum[i] = a_shared[i];
    }

    asm volatile("\n \
          v_fma_f32 %0, %1, %2, %0  \n \
          "
          :
          :"v"(wk_tile_m), "v"(wk_tile_n), "v"(local_id)
          );
#pragma unroll           
    for(int i=0; i <=64; i++){  
         asm volatile("\n \
          v_fma_f32 %0, %1, %2, %0  \n \
          "
          :
          :"v"(c[i]), "v"(c[i]), "v"(wk_tile_n)
          );
    }
    asm volatile ("\nds_read_b32 v2, v1\n");
    asm volatile ("\nv_mov_b32 v199, 0\n");
    asm volatile ("\ns_mov_b32 s17,  0\n");
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

  hipEventRecord(start, NULL);
  hipLaunchKernelGGL(sgemm_null, 
                dim3(M/128, N/128 ),
                dim3(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y),
                0, 0,
                deviceA ,deviceB ,deviceC, M, N, K, 0,0,0.0,0.0);

  hipEventRecord(stop, NULL);
  hipEventSynchronize(stop);

  hipEventElapsedTime(&eventMs, start, stop);

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
