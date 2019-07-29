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


#define NUM   4096

#define THREADS_PER_BLOCK_X  1
#define THREADS_PER_BLOCK_Y  1
#define THREADS_PER_BLOCK_Z  1

#define OUTER_LOOPS        10000
#define INNER_LOOPS        100



//local block size,  (256, 1)
//total threads (H * W, 1)
//




__global__ void
test_kernel(hipLaunchParm lp,
	int* __restrict__ buf, int mask, int outerLloops)
{

	int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
	__shared__ int ldsData[4096];
	ldsData[hipThreadIdx_x] = buf[x];


	int temp = hipThreadIdx_x;
	for(int i = 0; i < outerLloops; i++)	{
		for(int j = 0; j < INNER_LOOPS; j++)
		{
			temp = ldsData[temp] & mask;
		}
	}
	if (temp > 0)
	{
		buf[x] = temp;
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

	cout << "hip Device prop succeeded " << endl;

	hipEvent_t start, stop;

	hipEventCreate(&start);
	hipEventCreate(&stop);
	float eventMs = 1.0f;

	int i;
	int errors;

	hostA = (int*)malloc(NUM * sizeof(int));

	int* p;

	p = hostA;
	for (int i = 0; i < NUM; i += 1)
	{
		*p = i;
		p++;
	}

	HIP_ASSERT(hipMalloc((void**)& deviceA, NUM * sizeof(int)));
	HIP_ASSERT(hipMemcpy(deviceA, hostA, NUM * sizeof(int), hipMemcpyHostToDevice));

	hipLaunchKernel(test_kernel,
		dim3(1, 1, 1),
		dim3(1, 1, 1),
		0, 0,
		deviceA, 0, 1);


	hipEventRecord(start, NULL);
	hipLaunchKernel(test_kernel,
			dim3(1, 1, 1),
			dim3(1, 1, 1),
			0, 0,
			deviceA, 0, OUTER_LOOPS);
	
	hipEventRecord(stop, NULL);
	hipEventSynchronize(stop);

	hipEventElapsedTime(&eventMs, start, stop);

	printf("elapsed time:%f\n", eventMs);
	int latency = int(eventMs * (double)1.536*1e6 / ((double)INNER_LOOPS * (double)OUTER_LOOPS));
	printf("latency for Vega10(1.536Ghz):  %d cycles \n", latency );

	HIP_ASSERT(hipFree(deviceA));

	free(hostA);

	return errors;
}
