  
/*
Copyright (c) 2015-2016 Advanced Micro Devices, Inc. All rights reserved.
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/
#include <assert.h>
#include <stdio.h>
#include <algorithm>
#include <stdlib.h>
#include<iostream>
#include "hip/hip_runtime.h"


#define HIP_ASSERT(x) (assert((x)==hipSuccess))


#define WIDTH     (16384)
#define HEIGHT    (16384)

#define NUM       (WIDTH*HEIGHT)

#define THREADS_PER_BLOCK_X  16
#define THREADS_PER_BLOCK_Y  8
#define THREADS_PER_BLOCK_Z  1


__global__ void 
vectoradd_float_1d(float* __restrict__ a, const float* __restrict__ b, const float* __restrict__ c, int width, int height) 
 {   
#if 0       
       int n = 0;
       if((hipThreadIdx_x &0x20) == 0) 
       {
           n = ((hipThreadIdx_x &0x3e0 ) >> 1) | (hipThreadIdx_x & 0x1f);
       }

       int x = hipBlockDim_x * hipBlockIdx_x +  n ;  
#else        
       int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x ;     

       int n = x ;      
#endif        
       if ( n < (width * height)) {
             
             a[n] = b[n] + c[n];         
       }    
 }
__global__ void 
vectoradd_float_fill2_v1(float* __restrict__ a, const float* __restrict__ b, const float* __restrict__ c, int width, int height) 
{
      int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x  ;
      int i = x * 1;      
      float2* aa = (float2*)a;
      float2* bb = (float2*)b;
      float2* cc = (float2*)c;
      if ( i < (width * height)) {
          int n;
          for(n = i; (n+64*256 * 1) < (width * height); n+=64*256 * 1 )
          {   
             aa[n] = bb[n] + cc[n];                        
           //  aa[n+64*256] = bb[n+64*256] + cc[n+64*256];         
          }

          //last one
          if(n <  (width * height))
          {
              aa[n] = bb[n] + cc[n]; 
          }
      }
}

__global__ void 
vectoradd_float_fill4_v1(float4* __restrict__ a, const float4* __restrict__ b, const float4* __restrict__ c, int width, int height) 
{
      int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x  ;
      int i = x * 1;      
      if ( i < (width * height)) {
          int n;
          for(n = i; (n+64*256 * 4) < (width * height); n+=64*256 * 4 )
          {  
              for(int j=0; j < 4; j++) {
                    a[n+ j * 64 * 256] = b[n+ j * 64 * 256] + c[n + j * 64 * 256];       
              }            
          }

          //last 7
          for(int j = 0; j < 3; j++)
          {              
            if(n <  (width * height))
            {
                a[n] = b[n] + c[n]; 
            }
             n += 64 * 256;
          }
      }
}

__global__ void 
vectoradd_float_unroll_2_v1(float* __restrict__ a, const float* __restrict__ b, const float* __restrict__ c, int width, int height) 
 {        
      //int x = hipBlockDim_x * hipBlockIdx_x + ((hipThreadIdx_x + 224 * hipBlockIdx_x) &0x7f) ; 
      int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x  ;
      int i = x * 1;      
      if ( i < (width * height)) {
          int n;
          for(n = i; (n+64*256 * 2) < (width * height); n+=64*256 * 2 )
          {   
             a[n] = b[n] + c[n];                        
             a[n+64*256] = b[n+64*256] + c[n+64*256];         
          }

          //last one
          if(n <  (width * height))
          {
              a[n] = b[n] + c[n]; 
          }
      }
  }

__global__ void 
vectoradd_float_unroll_2_v2(float* __restrict__ a, const float* __restrict__ b, const float* __restrict__ c, int width, int height) 
 {        
      int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x ;     
      int i = x * 1;      
      if ( i < (width * height)) {
          int n;
          for(n = i; (n+32*256 * 2) < (width * height); n+=32*256 * 2 )
          {   
             a[n] = b[n] + c[n];         
             a[n+32*256] = b[n+32*256] + c[n+32*256];         
          }
          //last 
          if(n <  (width * height))
          {
              a[n] = b[n] + c[n]; 
          }
      }
  }

__global__ void 
vectoradd_float_unroll_2_v3(float* __restrict__ a, const float* __restrict__ b, const float* __restrict__ c, int width, int height) 
 {        
     //int x = hipBlockDim_x * hipBlockIdx_x + ((hipThreadIdx_x + 16 * hipBlockIdx_x) &0x7f) ; 
      int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x ;     
      int i = x * 1;      
      if ( i < (width * height)) {
          int n;
          for(n = i; (n+64*256 * 3) < (width * height); n+=64*256 * 3 )
          {   
             a[n] = b[n] + c[n];         
             a[n+64*256] = b[n+64*256] + c[n+64*256];         
             a[n+64*256*2] = b[n+64*256 *2 ] + c[n+64*256 *2];         
          }
          //last 2
          if(n <  (width * height))
          {
              a[n] = b[n] + c[n]; 
          }
          n +=  64 * 256;
          if(n <  (width * height))
          {
              a[n] = b[n] + c[n]; 
          }
      }
  }

__global__ void 
vectoradd_float(float* __restrict__ a, const float* __restrict__ b, const float* __restrict__ c, int width, int height) 
  {
 
      int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
      int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;

      int i = y * width  + x;      
      if ( i < (width * height)) {

        //for(int r=0; r < 8; r++) 
        {   
            int r = 0;
            a[i+width*r] = b[i+width*r] + c[i+width*r];
        }
      }
  }

#if 0
__kernel__ void vectoradd_float(float* a, const float* b, const float* c, int width, int height) {

  
  int x = blockDimX * blockIdx.x + threadIdx.x;
  int y = blockDimY * blockIdy.y + threadIdx.y;

  int i = y * width + x;
  if ( i < (width * height)) {
    a[i] = b[i] + c[i];
  }
}
#endif

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
    hostB[i] = (float)i;
    hostC[i] = (float)i*100.0f;
  }
  
  HIP_ASSERT(hipMalloc((void**)&deviceA, NUM * sizeof(float)));
  HIP_ASSERT(hipMalloc((void**)&deviceB, NUM * sizeof(float)));
  HIP_ASSERT(hipMalloc((void**)&deviceC, NUM * sizeof(float)));
  
  HIP_ASSERT(hipMemcpy(deviceB, hostB, NUM*sizeof(float), hipMemcpyHostToDevice));
  HIP_ASSERT(hipMemcpy(deviceC, hostC, NUM*sizeof(float), hipMemcpyHostToDevice));

   
  {
  hipLaunchKernelGGL(vectoradd_float, 
                  dim3(WIDTH/THREADS_PER_BLOCK_X, HEIGHT/THREADS_PER_BLOCK_Y),
                  dim3(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y),
                  0, 0,
                  deviceA ,deviceB ,deviceC ,WIDTH ,HEIGHT);

  HIP_ASSERT(hipMemcpy(hostA, deviceA, NUM*sizeof(float), hipMemcpyDeviceToHost));
  hipEvent_t start, stop; 
  hipEventCreate(&start);
  hipEventCreate(&stop);
  float eventMs = 0.0f;

    hipEventRecord(start, NULL);
    hipLaunchKernelGGL(vectoradd_float, 
            dim3(WIDTH/THREADS_PER_BLOCK_X, HEIGHT/THREADS_PER_BLOCK_Y),
            dim3(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y),
            0, 0,
            deviceA ,deviceB ,deviceC ,WIDTH ,HEIGHT);


    hipEventRecord(stop, NULL);
    hipEventSynchronize(stop);

    hipEventElapsedTime(&eventMs, start, stop);
    float mem_bw = (float) WIDTH * (float) HEIGHT * sizeof(float) * 3 /1024/1024;
    mem_bw = mem_bw / eventMs;

    printf("vec add 2D tiling: %f GB/s \n",mem_bw);
  }


  // verify the results
  errors = 0;
  for (i = 0; i < NUM; i++) {
    if (hostA[i] != (hostB[i] + hostC[i])) {
      errors++;
    }
  }
  if (errors!=0) {
    printf("FAILED: %d errors\n",errors);
  } else {
      printf ("PASSED!\n");
  }
  for(int i = 1024; i <= 16384; i *=2)
  {
        hipEvent_t start, stop; 
        hipLaunchKernelGGL(vectoradd_float_1d, 
                    dim3((i*i)/1024),
                    dim3(1024),
                    0, 0,
                    deviceA ,deviceB ,deviceC ,i ,i);

//  HIP_ASSERT(hipMemcpy(hostA, deviceA, NUM*sizeof(float), hipMemcpyDeviceToHost));

        hipEventCreate(&start);
        hipEventCreate(&stop);
        float eventMs = 0.0f;

        hipEventRecord(start, NULL);
        hipLaunchKernelGGL(vectoradd_float_1d, 
                    dim3((i*i)/1024),
                    dim3(1024),
                    0, 0,
                    deviceA ,deviceB ,deviceC ,i ,i);


        hipEventRecord(stop, NULL);
        hipEventSynchronize(stop);

        hipEventElapsedTime(&eventMs, start, stop);
        float mem_bw = (float) i * (float) i * sizeof(float) * 3 /1024/1024;
        mem_bw = mem_bw / eventMs;

        printf("vectoradd_float_1d [%dx%d], %f, %f GB/s \n",i, i, eventMs, mem_bw);

  }

  for(int i = 1024; i <= 16384; i *=2)
  {
        hipEvent_t start, stop; 
        hipLaunchKernelGGL(vectoradd_float_fill2_v1, 
                    dim3(64, 1),
                    dim3(256),
                    0, 0,
                    deviceA ,deviceB ,deviceC ,i/2 ,i);

//  HIP_ASSERT(hipMemcpy(hostA, deviceA, NUM*sizeof(float), hipMemcpyDeviceToHost));

        hipEventCreate(&start);
        hipEventCreate(&stop);
        float eventMs = 0.0f;

        hipEventRecord(start, NULL);
        hipLaunchKernelGGL(vectoradd_float_fill2_v1, 
                    dim3(64, 1),
                    dim3(256),
                    0, 0,
                    deviceA ,deviceB ,deviceC ,i/2 ,i);


        hipEventRecord(stop, NULL);
        hipEventSynchronize(stop);

        hipEventElapsedTime(&eventMs, start, stop);
        float mem_bw = (float) i * (float) i * sizeof(float) * 3 /1024/1024;
        mem_bw = mem_bw / eventMs;

        printf("vectoradd_float_fill2_v1 [%dx%d], %f, %f GB/s \n",i,i, eventMs, mem_bw);

  }

  for(int i = 1024; i <= 16384; i *=2)
  {
        hipEvent_t start, stop; 
        hipLaunchKernelGGL(vectoradd_float_unroll_2_v1, 
                    dim3(64, 1),
                    dim3(256),
                    0, 0,
                    deviceA ,deviceB ,deviceC ,i ,i);

//  HIP_ASSERT(hipMemcpy(hostA, deviceA, NUM*sizeof(float), hipMemcpyDeviceToHost));

        hipEventCreate(&start);
        hipEventCreate(&stop);
        float eventMs = 0.0f;

        hipEventRecord(start, NULL);
        hipLaunchKernelGGL(vectoradd_float_unroll_2_v1, 
                    dim3(64, 1),
                    dim3(256),
                    0, 0,
                    deviceA ,deviceB ,deviceC ,i ,i);


        hipEventRecord(stop, NULL);
        hipEventSynchronize(stop);

        hipEventElapsedTime(&eventMs, start, stop);
        float mem_bw = (float) i * (float) i * sizeof(float) * 3 /1024/1024;
        mem_bw = mem_bw / eventMs;

        printf("vectoradd_float_unroll_2_v1 [%dx%d], %f, %f GB/s \n",i,i, eventMs, mem_bw);

  }

  for(int i = 1024; i <= 16384; i *=2)
  {
        hipEvent_t start, stop; 
        hipLaunchKernelGGL(vectoradd_float_unroll_2_v2, 
                    dim3(32, 1),
                    dim3(1024),
                    0, 0,
                    deviceA ,deviceB ,deviceC ,i ,i);

//  HIP_ASSERT(hipMemcpy(hostA, deviceA, NUM*sizeof(float), hipMemcpyDeviceToHost));

        hipEventCreate(&start);
        hipEventCreate(&stop);
        float eventMs = 0.0f;

        hipEventRecord(start, NULL);
        hipLaunchKernelGGL(vectoradd_float_unroll_2_v2, 
                    dim3(32, 1),
                    dim3(256),
                    0, 0,
                    deviceA ,deviceB ,deviceC ,i ,i);


        hipEventRecord(stop, NULL);
        hipEventSynchronize(stop);

        hipEventElapsedTime(&eventMs, start, stop);
        float mem_bw = (float) i * (float) i * sizeof(float) * 3 /1024/1024;
        mem_bw = mem_bw / eventMs;

        printf(" vectoradd_float_unroll_2_v2 [%dx%d], %f, %f GB/s \n",i, i, eventMs, mem_bw);
  }

  for(int i = 1024; i <= 16384; i *=2)
  {
        hipEvent_t start, stop; 
        hipLaunchKernelGGL(vectoradd_float_unroll_2_v3, 
                    dim3(64, 1),
                    dim3(256),
                    0, 0,
                    deviceA ,deviceB ,deviceC ,i ,i);

//  HIP_ASSERT(hipMemcpy(hostA, deviceA, NUM*sizeof(float), hipMemcpyDeviceToHost));

        hipEventCreate(&start);
        hipEventCreate(&stop);
        float eventMs = 0.0f;

        hipEventRecord(start, NULL);
        hipLaunchKernelGGL(vectoradd_float_unroll_2_v3, 
                    dim3(64, 1),
                    dim3(256),
                    0, 0,
                    deviceA ,deviceB ,deviceC ,i ,i);


        hipEventRecord(stop, NULL);
        hipEventSynchronize(stop);

        hipEventElapsedTime(&eventMs, start, stop);
        float mem_bw = (float) i * (float) i * sizeof(float) * 3 /1024/1024;
        mem_bw = mem_bw / eventMs;

        printf(" vectoradd_float_unroll_2_v3 [%dx%d], %f, %f GB/s \n",i, i, eventMs, mem_bw);
  }

  HIP_ASSERT(hipFree(deviceA));
  HIP_ASSERT(hipFree(deviceB));
  HIP_ASSERT(hipFree(deviceC));

  free(hostA);
  free(hostB);
  free(hostC);

  //hipResetDefaultAccelerator();

  return errors;
}
