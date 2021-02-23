//hipcc --amdgpu-target=gfx900 sgemm128x128.cpp  -o sgemm
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
template<uint32_t offset>
inline __device__ void global_load(float* ptr, Float4 &val) {
    if(offset == 0) {
    asm volatile("\n \
    global_load_dwordx4 %0, %1, off \n \
    "
    :"=v"(val)
    :"v"(ptr)
    );
    return;
    }
    if(offset == 8) {
    asm volatile("\n \
    global_load_dwordx4 %0, %1, off offset:32 \n \
    "
    :"=v"(val)
    :"v"(ptr));
    }
}


template<uint32_t cnt>
inline __device__ void lgkmcnt(){
  if(cnt == 0) {
    asm volatile("\n \
    s_waitcnt lgkmcnt(0) \n \
    "::);
  }
  if(cnt == 1) {
    asm volatile("\n \
    s_waitcnt lgkmcnt(1) \n \
    "::);
  }
  if(cnt == 2) {
    asm volatile("\n \
    s_waitcnt lgkmcnt(2) \n \
    "::);
  }
  if(cnt == 3) {
    asm volatile("\n \
    s_waitcnt lgkmcnt(3) \n \
    "::);
  }
  if(cnt == 4) {
    asm volatile("\n \
    s_waitcnt lgkmcnt(4) \n \
    "::);
  }
  if(cnt == 5) {
    asm volatile("\n \
    s_waitcnt lgkmcnt(5) \n \
    "::);
  }
  if(cnt == 6) {
    asm volatile("\n \
    s_waitcnt lgkmcnt(6) \n \
    "::);
  }

/**
* Disabling as 16 is to high to fit in 4bits (15 max)
  if(cnt == 16) {
    asm volatile("\n \
    s_waitcnt lgkmcnt(16) \n \
    "::);
  }
*/
}

template<uint32_t cnt>
inline __device__ void vmcnt() {
    if(cnt == 0) {
      asm volatile ("\n \
      s_waitcnt vmcnt(0) \n \
      "::);
    }
    if(cnt == 1) {
      asm volatile ("\n \
      s_waitcnt vmcnt(1) \n \
      "::);
    }
    if(cnt == 2) {
      asm volatile ("\n \
      s_waitcnt vmcnt(2) \n \
      "::);
    }
    if(cnt == 4) {
      asm volatile ("\n \
      s_waitcnt vmcnt(2) \n \
      "::);
    }
}


inline __device__ void Matrix4x1X4(float* a,  float* b, float* c, int a_id, int b_id){  
      for(int i = a_id; i < a_id+4; i++)
      for(int j = b_id; j < b_id+4; j++)
      {
          asm volatile("\n \
          v_fma_f32 %0, %1, %2, %0  \n \
          "
          :
          :"v"(c[i*8+j]), "v"(a[i]), "v"(b[j])
          );

      }
}

inline __device__ void Matrix8x1X8(float* a,  float* b, float* c){  
    Matrix4x1X4(a,b,c,0,0);
    Matrix4x1X4(a,b,c,4,0);
    Matrix4x1X4(a,b,c,0,4);
    Matrix4x1X4(a,b,c,4,4);
}


//MxK=128x1 will stored as following in Shared memory 
//0-3, 8-11, 16-19, ..., 120-123, 4-7,12-15,20-23,...,124-127
#define UNROLL_SIZE 8
__global__ void sgemm_nt_128x128_lds_unroll_8_scheduling(const float* a, const float* b, float* __restrict__ c, const int m, const int n, const int k ){
  int wk_tile_m =  hipBlockIdx_y * 128 ;
  int wk_tile_n =  hipBlockIdx_x * 128 ;
  int thread_tile_m = wk_tile_m + hipThreadIdx_y * 8;
  int thread_tile_n = wk_tile_n + hipThreadIdx_x * 8;
  int local_id = hipThreadIdx_y * 16 + hipThreadIdx_x;
  int local_tile_m, local_tile_n;
  local_tile_m = hipThreadIdx_x * 4;  //0-3, 8-11, 16-19, ..., 120-123, 4-7,12-15,20-23,...,124-127
  local_tile_n = hipThreadIdx_y * 4;

  
  __shared__ float a_shared[128*UNROLL_SIZE];
  __shared__ float b_shared[128*UNROLL_SIZE];
 
  float* a_ptr = NULL;
  float* b_ptr = NULL;
  int lds_write = 0;
  //every 256 threads Load A 256x4 
  //every 256 threads Load B 256x4 

  //Continuous 2X threads Load 8x DWORDs
  a_ptr = (float*)(a + (wk_tile_m + (local_id >> 1)) * k + (local_id &0x1) * 4); 
  b_ptr = (float*)(b + (wk_tile_n + (local_id >> 1)) * k + (local_id &0x1) * 4);


  //LDS Stores 128
  {
      int read_offset = local_id >> 1;
      //m read_offset, 0, 0, 1  1 2 2 3 3 4  4  5   5   6   6  7  7 
      // lds_write     0  0  1  1 2 2 3 3 64 64 65  65  66  67 ...

      //High bits[6:3] * 4 + low  2bits & 0x3 //Clamp to 64       
      //lds_write =  (( read_offset >> 3 ) << 2 )  +  (read_offset & 0x3);
      lds_write =  (( read_offset & 0x78 ) >>1)  +  (read_offset & 0x3);
      //if ((read_offset & 0x4) == 0x4)
      {
          // 64 +  
          lds_write = lds_write | ((read_offset & 0x4 ) << 4);
      }
      lds_write = lds_write + (local_id & 0x1) * 128 * 4;
  }

  //PreLoad 128*8 DWORDs
  Float4 readA;
  Float4 readB;

  //for(int i=0; i < 4; i++)  
  //{
  //  g_a[i] = a_ptr[i];
  //  g_b[i] = b_ptr[i];
  //}

   lgkmcnt<0>();
   float sum[64];  
   // for(int i=0; i < 64 i++) sum[i] = 0  will have allocate 1 VGPR only
   // Following code to load SUM from LDS with different vlaue, SUM will allocate 64 Values
   // In short: SUM is initialized with 1 same value will have only 1 VGPR allocations 
    a_shared[local_id] = (local_id >> 6) * 1.0f;
    __syncthreads();
    for(int i=0; i <=64; i++){
        sum[i] = a_shared[i];
    }


  global_load<0>(a_ptr, readA);
  global_load<0>(b_ptr, readB);  

    vmcnt<0>();
     a_shared[lds_write] = readA.x;
     a_shared[lds_write + 128] = readA.y;
     a_shared[lds_write + 256] = readA.z;
     a_shared[lds_write + 384] = readA.w;    

     b_shared[lds_write] = readB.x;
     b_shared[lds_write + 128] = readB.y;
     b_shared[lds_write + 256] = readB.z;
     b_shared[lds_write + 384] = readB.w;

    float adata[8];
    float bdata[8];
    float adata2[8];
    float bdata2[8];

  //unroll 
  for(int kk=0; kk < k; kk+=UNROLL_SIZE) {   
     asm volatile("\n  s_barrier\n"::);      
     if( (kk + UNROLL_SIZE) < k)   {
          a_ptr += 8;
          b_ptr += 8;
          global_load<0>(a_ptr, readA);
          global_load<0>(b_ptr, readB);      
     }    
     
     lgkmcnt<0>();  
    
#pragma unroll
     for(int  s =0; s < 1; s++)
     {     
 
          s = 0;
          for(int t=0; t < 4; t++){
            adata[t] = a_shared[local_tile_m + t + s * 128];
            bdata[t] = b_shared[local_tile_n + t + s * 128];
          }
          for(int t=4; t < 8; t++){
            adata[t] = a_shared[local_tile_m + t + 64 + s * 128];
            bdata[t] = b_shared[local_tile_n + t + 64 + s * 128];
          }
          
          s = 1 ;
          for(int t=0; t < 4; t++){
            adata2[t] = a_shared[local_tile_m + t + s * 128];
            bdata2[t] = b_shared[local_tile_n + t + s * 128];
          }
          for(int t=4; t < 8; t++){
            adata2[t] = a_shared[local_tile_m + t + 64 + s * 128];
            bdata2[t] = b_shared[local_tile_n + t + 64 + s * 128];
          }
          //lgkmcnt<4>();
          Matrix8x1X8(&adata[0],&bdata[0],&sum[0]);
          

          s = 2 ;
          for(int t=0; t < 4; t++){
            adata[t] = a_shared[local_tile_m + t + s * 128];
            bdata[t] = b_shared[local_tile_n + t + s * 128];
          }
          for(int t=4; t < 8; t++){
            adata[t] = a_shared[local_tile_m + t + 64 + s * 128];
            bdata[t] = b_shared[local_tile_n + t + 64 + s * 128];
          }
          //lgkmcnt<4>();
          Matrix8x1X8(&adata2[0],&bdata2[0],&sum[0]);
          s = 3;
          for(int t=0; t < 4; t++){
            adata2[t] = a_shared[local_tile_m + t + s * 128];
            bdata2[t] = b_shared[local_tile_n + t + s * 128];
          }
          for(int t=4; t < 8; t++){
            adata2[t] = a_shared[local_tile_m + t + 64 + s * 128];
            bdata2[t] = b_shared[local_tile_n + t + 64 + s * 128];
          }
          //lgkmcnt<4>();
          Matrix8x1X8(&adata[0],&bdata[0],&sum[0]); 

          s = 4 ;
          for(int t=0; t < 4; t++){
            adata[t] = a_shared[local_tile_m + t + s * 128];
            bdata[t] = b_shared[local_tile_n + t + s * 128];
          }
          for(int t=4; t < 8; t++){
            adata[t] = a_shared[local_tile_m + t + 64 + s * 128];
            bdata[t] = b_shared[local_tile_n + t + 64 + s * 128];
          }
          //lgkmcnt<4>();
          Matrix8x1X8(&adata2[0],&bdata2[0],&sum[0]);
          s = 5;
          for(int t=0; t < 4; t++){
            adata2[t] = a_shared[local_tile_m + t + s * 128];
            bdata2[t] = b_shared[local_tile_n + t + s * 128];
          }
          for(int t=4; t < 8; t++){
            adata2[t] = a_shared[local_tile_m + t + 64 + s * 128];
            bdata2[t] = b_shared[local_tile_n + t + 64 + s * 128];
          }
          //lgkmcnt<4>();
          Matrix8x1X8(&adata[0],&bdata[0],&sum[0]);

          s = 6 ;
          for(int t=0; t < 4; t++){
            adata[t] = a_shared[local_tile_m + t + s * 128];
            bdata[t] = b_shared[local_tile_n + t + s * 128];
          }
          for(int t=4; t < 8; t++){
            adata[t] = a_shared[local_tile_m + t + 64 + s * 128];
            bdata[t] = b_shared[local_tile_n + t + 64 + s * 128];
          }
          //lgkmcnt<4>();
          Matrix8x1X8(&adata2[0],&bdata2[0],&sum[0]);

          s = 7;
          for(int t=0; t < 4; t++){
            adata2[t] = a_shared[local_tile_m + t + s * 128];
            bdata2[t] = b_shared[local_tile_n + t + s * 128];
          }
          for(int t=4; t < 8; t++){
            adata2[t] = a_shared[local_tile_m + t + 64 + s * 128];
            bdata2[t] = b_shared[local_tile_n + t + 64 + s * 128];
          }
          //lgkmcnt<4>();
          Matrix8x1X8(&adata[0],&bdata[0],&sum[0]);

          vmcnt<0>();
          a_shared[lds_write] = readA.x;
          a_shared[lds_write + 128] = readA.y;
          a_shared[lds_write + 256] = readA.z;
          a_shared[lds_write + 384] = readA.w;    

          b_shared[lds_write] = readB.x;
          b_shared[lds_write + 128] = readB.y;
          b_shared[lds_write + 256] = readB.z;
          b_shared[lds_write + 384] = readB.w;
          //lgkmcnt<4>();
          Matrix8x1X8(&adata2[0],&bdata2[0],&sum[0]);
      
    }
  }

  
  //store   
  for(int i=0; i < 8; i++)
    for(int j=0; j < 8; j++){
      c[ (thread_tile_m + i) * n + thread_tile_n + j] = sum[i*8+j ];
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
          hipLaunchKernelGGL(sgemm_nt_128x128_lds_unroll_8_scheduling, 
                        dim3(mnk/128, mnk/128 ),
                        dim3(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y),
                        0, 0,
                        deviceA ,deviceB ,deviceC, mnk, mnk, mnk);

          hipEventRecord(start, NULL);
        for (int i = 0; i < 1; i++)
        {
          hipLaunchKernelGGL(sgemm_nt_128x128_lds_unroll_8_scheduling, 
                        dim3(mnk/128, mnk/128 ),
                        dim3(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y),
                        0, 0,
                        deviceA ,deviceB ,deviceC, mnk, mnk, mnk);
        }

          hipEventRecord(stop, NULL);
          hipEventSynchronize(stop);

          hipEventElapsedTime(&eventMs, start, stop);
   
		      //printf("elapsed time:%f\n", eventMs);
          double ips = ( double)(mnk)*( double)mnk*( double)mnk /1024/1024/1024*1;
		      ips = ips / ( double)eventMs * 1000 ;
		      printf("sgemm_nt_128x128_lds_unroll16_scheduling:[%d x %d % d ] ==> %lf G FMAs/s, ms: %f\n", mnk,mnk,mnk, ips, eventMs);
          usleep (100 *1000);

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
