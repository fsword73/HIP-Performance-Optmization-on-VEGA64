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

#define NUM  (1024*1024*256) 

#define TOTAL_THREADS (1024*1024*256) 

#define THREADS_PER_BLOCK_X  512
#define THREADS_PER_BLOCK_Y  1
#define THREADS_PER_BLOCK_Z  1

#define CACHE_LINE_SIZE    64

__global__ void 
test_kernel(hipLaunchParm lp,
	int* __restrict__ buf, int stride )
{

  	int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;  
	x = (x * stride) & (NUM-1);	

    int t = buf[x];    
    
  	if( t > 0x3fffffff)
  	{
  		buf[x] = t+1; 
  	}
}





using namespace std;

int main() {
  
  int* hostA;

  int* deviceA;

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

  hostA = (int*)malloc(NUM * sizeof(int));
  
  
  int* p;
  
  p = hostA;  
  for(int i=0; i< NUM; i+=1)
  {  
		*p = 1;
		p++; 
  }
  
  
  HIP_ASSERT(hipMalloc((void**)&deviceA, NUM * sizeof(int)));
  HIP_ASSERT(hipMemcpy(deviceA, hostA, NUM * sizeof(int), hipMemcpyHostToDevice));

  hipLaunchKernel(test_kernel,
                  dim3(1, 1,1),
                  dim3(THREADS_PER_BLOCK_X, 1, 1),
	              0, 0,
                  deviceA, 1);

  for (int i = 1; i <= CACHE_LINE_SIZE; i = i << 1) {
	  hipEventRecord(start, NULL);
	  hipLaunchKernel(test_kernel,
		  dim3(NUM/THREADS_PER_BLOCK_X, 1,1),
		  dim3(THREADS_PER_BLOCK_X, 1, 1),
		  0, 0,
		  deviceA, i);

	  hipEventRecord(stop, NULL);
	  hipEventSynchronize(stop);

	  hipEventElapsedTime(&eventMs, start, stop);

	  printf("RangeSize[%8d], kernel_time (hipEventElapsedTime) =%6.3fms\n", i * sizeof(int), eventMs);
  }

  HIP_ASSERT(hipFree(deviceA));

  free(hostA);

  return errors;
}
