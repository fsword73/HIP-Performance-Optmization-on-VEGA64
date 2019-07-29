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


#define NUM   (256*1024*1024)

#define THREADS_PER_BLOCK_X  256
#define THREADS_PER_BLOCK_Y  1
#define THREADS_PER_BLOCK_Z  1



__global__ void
test_kernel(hipLaunchParm lp,
	int* __restrict__ buf, int reduce_number_once)
{

	int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;

	__shared__ int ldsData[256];
	ldsData[hipThreadIdx_x] = buf[x];
	__syncthreads();
	
	int sum =0;
	if (reduce_number_once == 2)
	{
		for (int s = 256 >> 1; s > 0; s = s >> 1)
		{
			if (s > hipThreadIdx_x) {
				ldsData[hipThreadIdx_x] = ldsData[hipThreadIdx_x] + ldsData[hipThreadIdx_x + s];
			}
			__syncthreads();
		}
		if (hipThreadIdx_x == 0)
		{
			sum += ldsData[0];
		}
			
		
	}
	if (reduce_number_once == 4)
	{
			for (int s = 256 >> 2; s > 0; s = s >> 2)
			{
				if (s > hipThreadIdx_x) {
					ldsData[hipThreadIdx_x] = ldsData[hipThreadIdx_x] + ldsData[hipThreadIdx_x + s] +
						ldsData[hipThreadIdx_x + 2 * s] + ldsData[hipThreadIdx_x + 3 * s];
				}
			}
			if (hipThreadIdx_x == 0)
			{
				sum += ldsData[0];
			}
	}

	if ((hipThreadIdx_x == 0) && sum > 9999)
	{
		buf[hipBlockIdx_x] = sum;
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
	for (int i = 0; i < NUM; i++)
	{
		p[i] = 0;
	}

	HIP_ASSERT(hipMalloc((void**)& deviceA, NUM * sizeof(int)));
	HIP_ASSERT(hipMemcpy(deviceA, hostA, NUM * sizeof(int), hipMemcpyHostToDevice));


	hipLaunchKernel(test_kernel,
		dim3(1, 1, 1),
		dim3(THREADS_PER_BLOCK_X, 1, 1),
		0, 0,
		deviceA, 2);


		hipEventRecord(start, NULL);
		hipLaunchKernel(test_kernel,
			dim3(NUM/ THREADS_PER_BLOCK_X, 1, 1),
			dim3(THREADS_PER_BLOCK_X, 1,1),
			0, 0,
			deviceA, 2);

		hipEventRecord(stop, NULL);
		hipEventSynchronize(stop);

		hipEventElapsedTime(&eventMs, start, stop);

		printf("Reduce 2 once:  elapsed time:%f\n", eventMs);

		hipEventRecord(start, NULL);
		hipLaunchKernel(test_kernel,
			dim3(NUM / THREADS_PER_BLOCK_X, 1, 1),
			dim3(THREADS_PER_BLOCK_X, 1, 1),
			0, 0,
			deviceA, 4);

		hipEventRecord(stop, NULL);
		hipEventSynchronize(stop);

		hipEventElapsedTime(&eventMs, start, stop);

		printf("Reduce 4 once:  elapsed time:%f\n", eventMs);

	HIP_ASSERT(hipFree(deviceA));

	free(hostA);

	return errors;
}
