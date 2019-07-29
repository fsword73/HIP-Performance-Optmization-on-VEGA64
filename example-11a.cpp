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

#define THREADS_PER_BLOCK_X  1
#define THREADS_PER_BLOCK_Y  1
#define THREADS_PER_BLOCK_Z  1



#define INNER_LOOPS  100
#define OUTER_LOOPS  10000

__global__ void
test_kernel(hipLaunchParm lp,
	int* __restrict__ buf, int outerLoops)
{

	int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;

	double f = sin(x / 256.0);
	for (int i = 0; i < outerLoops; i++)
		for (int j = 0; j < INNER_LOOPS;j++)
			f = sin(f);
	if (f > 0.999)
	{
		buf[x] = f;
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

	HIP_ASSERT(hipMalloc((void**)& deviceA, NUM * sizeof(int)));
	HIP_ASSERT(hipMemcpy(deviceA, hostA, NUM * sizeof(int), hipMemcpyHostToDevice));

	hipLaunchKernel(test_kernel,
		dim3(1, 1, 1),
		dim3(1, 1, 1),
		0, 0,
		deviceA, 0x0);

		hipEventRecord(start, NULL);
		hipLaunchKernel(test_kernel,
			dim3(1, 1, 1),
			dim3(1, 1, 1),
			0, 0,
			deviceA, OUTER_LOOPS);

		hipEventRecord(stop, NULL);
		hipEventSynchronize(stop);

		hipEventElapsedTime(&eventMs, start, stop);

		printf("elapsed time:%f\n", eventMs);
		double cycles = (double)eventMs * (double)1.536 * (double)1e6 / OUTER_LOOPS/INNER_LOOPS;
		printf("sin --double needs %d cycles\n", (int)cycles);
	

	HIP_ASSERT(hipFree(deviceA));

	free(hostA);

	return errors;
}
