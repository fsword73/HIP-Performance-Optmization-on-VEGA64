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

#define MAX_BLOCKS           1024
#define THREADS_PER_BLOCK_X  256
#define THREADS_PER_BLOCK_Y  1
#define THREADS_PER_BLOCK_Z  1
#define FMA_PER_THREADS       1000000

__global__ void 
null_kernel(hipLaunchParm lp,
	float* __restrict__ a)
{

	int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
	int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;	
	float t0 = (float)x / (float) (x + 1);
	float t1 = float(y + 1) / (float)(y + 100000000);

	float sum=0.0;

	for(int i =0; i < FMA_PER_THREADS/100;i++)
	{
		for(int j=0; j < 100; j++)
		sum = t0 *sum + t1;
	}
  
	if( (float(x)+sum) < -1.0f)
	{
		a[0] = sum; 
	}
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
  

  hipLaunchKernel(null_kernel,
                  dim3(1, 1),
                  dim3(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y, THREADS_PER_BLOCK_Z),
	              0, 0,
                  deviceA);

  for (int i = 1; i < 5; i = i + 1) {
	  hipEventRecord(start, NULL);
	  hipLaunchKernel(null_kernel,
		  dim3(1, 1,1),
		  dim3(THREADS_PER_BLOCK_X*i, 1, 1),
		  0, 0,
		  deviceA);

	  hipEventRecord(stop, NULL);
	  hipEventSynchronize(stop);

	  hipEventElapsedTime(&eventMs, start, stop);

	  printf("kernel_time (hipEventElapsedTime) =%6.3fms\n", eventMs);
	  double FMA_per_cycle = double(THREADS_PER_BLOCK_X) * i * double(FMA_PER_THREADS) / eventMs / (1.536 * 1e6) + 0.5;

	  printf("Total Threads = %d * %d, FMA_per_cycle for Vega10 - 1.536GHz = %6d\n", 1, THREADS_PER_BLOCK_X * i, (int)FMA_per_cycle);
  }

 

  
  HIP_ASSERT(hipFree(deviceA));

  free(hostA);

  return errors;
}
