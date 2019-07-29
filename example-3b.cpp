/*
please note that the series of optmiztion technology is not in official document.  

All the tests are based on AMD MI25 radeon instict and AMD ROCm. 
*/


#include <assert.h>
#include <stdio.h>
#include <algorithm>
#include <stdlib.h>
#include<iostream>
#include "hip/hip_runtime.h"



#define HIP_ASSERT(x) (assert((x)==hipSuccess))

#define NUM  1 


#define THREADS_PER_BLOCK_X  256
#define THREADS_PER_BLOCK_Y  1
#define THREADS_PER_BLOCK_Z  1
#define MAX_VGPRS            1024
#define MAX_VGPRS			  255

__global__ void
test_kernel_255(hipLaunchParm lp,
	float* __restrict__ a)
{
  asm volatile("v_mov_b32 v0, 0");
	asm volatile("v_mov_b32 v255, 0" );
}


__global__ void
test_kernel_256(hipLaunchParm lp,
	float* __restrict__ a)
{
  asm volatile("v_mov_b32 v0, 0");
	asm volatile("v_mov_b32 v256, 0");
}


using namespace std;

int main() {
  
  float* hostA;

  float* deviceA;

  hipDeviceProp_t devProp;
  hipGetDeviceProperties(&devProp, 0);
  cout << " System minor " << devProp.minor << endl;
  cout << " System major " << devProp.major << endl;
  cout << " agent prop name " << devProp.name << endl;

  cout << "hip Device prop succeeded " << endl ;

  hipEvent_t start, stop;

  hipEventCreate(&start);
  hipEventCreate(&stop);
  float eventMs = 1.0f;

  int i;
  int errors;

  hostA = (float*)malloc(NUM * sizeof(float));
  

  
  HIP_ASSERT(hipMalloc((void**)&deviceA, NUM * sizeof(float)));
  


  int vgprs;
  vgprs = 255;
  printf(" Begin lauch %d VGPRs passed \n", vgprs);
  hipLaunchKernel(test_kernel_255,
	dim3(1, 1),
	dim3(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y, THREADS_PER_BLOCK_Z),
	0, 0,
	deviceA);

  printf("%d VGPRs passed \n", vgprs);
  



  
  
  HIP_ASSERT(hipFree(deviceA));

  free(hostA);

  return errors;
}
