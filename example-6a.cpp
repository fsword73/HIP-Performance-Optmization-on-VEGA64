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
#define MEM_STRIDE  256 

#define THREADS_PER_BLOCK_X  1
#define THREADS_PER_BLOCK_Y  1
#define THREADS_PER_BLOCK_Z  1

#define  MAX_MEM_READS (1024*1024)

__global__ void 
test_kernel(hipLaunchParm lp,
	int* __restrict__ buf)
{

  	int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
  
    int t = buf[x];
    
    //dependency reads
    for( int i=1; i < MAX_MEM_READS; i++)
    {
       t = buf[t * i ];
    }   		
    
  	if( t > 0x3fffffff)
  	{
  		buf[x] = t; 
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
		*p = 256;
		p++; 
  }
  
  
  HIP_ASSERT(hipMalloc((void**)&deviceA, NUM * sizeof(int)));
  HIP_ASSERT(hipMemcpy(deviceA, hostA, NUM * sizeof(int), hipMemcpyHostToDevice));

  hipLaunchKernel(test_kernel,
                  dim3(1, 1,1),
                  dim3(1, 1, 1),
	              0, 0,
                  deviceA);

  for (int i = 1; i < 2; i = i + 1) {
	  hipEventRecord(start, NULL);
	  hipLaunchKernel(test_kernel,
		  dim3(1, 1,1),
		  dim3(1, 1, 1),
		  0, 0,
		  deviceA);

	  hipEventRecord(stop, NULL);
	  hipEventSynchronize(stop);

	  hipEventElapsedTime(&eventMs, start, stop);

	  printf("kernel_time (hipEventElapsedTime) =%6.3fms\n", eventMs);
	  double mem_read_latency_cycle = double(eventMs) * double(1.536 * 1e6 ) / double(MAX_MEM_READS) ;

	  printf("mem_read_latency_cycle = %5d cycles for Vega10--1.536GHz\n", (int)mem_read_latency_cycle);
  }

  HIP_ASSERT(hipFree(deviceA));

  free(hostA);

  return errors;
}
