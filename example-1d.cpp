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
#define TOTAL_THREADS  (1024*1024*1024)
#define NUM  1 


#define THREADS_PER_BLOCK_X  1
#define THREADS_PER_BLOCK_Y  1
#define THREADS_PER_BLOCK_Z  1
#define MAX_THREADS  10  


__global__ void 
null_kernel(hipLaunchParm lp,
	float* __restrict__ a)
{

	//int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
	//int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;	

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
                  dim3(1, 1,1),
                  dim3(2, 2, 1),
	              0, 0,
                  deviceA);

  for (int i = 1; i < (MAX_THREADS + 1); i++) {
	  hipEventRecord(start, NULL);
	  hipLaunchKernel(null_kernel,
		  dim3(TOTAL_THREADS / (i*i*i), 1,1),
		  dim3(i, i, i),
		  0, 0,
		  deviceA);

	  hipEventRecord(stop, NULL);
	  hipEventSynchronize(stop);

	  hipEventElapsedTime(&eventMs, start, stop);

	  printf("kernel_time (hipEventElapsedTime) =%6.3fms\n", eventMs);
	  printf("threads_per_block = [%d,%d,%d],Threads_per_cycle for Vega10 - 1.536GHz = % 3d\n", i,i,i, int(TOTAL_THREADS / eventMs / 1.536 / 1e6+ 0.5));
  }

  
  HIP_ASSERT(hipFree(deviceA));

  free(hostA);

  return errors;
}
