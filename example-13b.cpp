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


#define N   1024
#define C   1024
#define H   7
#define W   7
#define PADDING   3

#define H_NEW  13
#define W_NEW  13 

#define NUM   (N * C * H_NEW * W_NEW)
#define RW_PER_THREAD 16

__global__ void
test_kernel(hipLaunchParm lp,
	float* __restrict__ bufA, float* __restrict__ bufB )
{

	int hw =  hipThreadIdx_x;
	int cc = RW_PER_THREAD * hipBlockIdx_y;
	int n = hipBlockIdx_z;
	float org_data[16];
	if (hw < (H_NEW * W_NEW))
	{

		int hh = hw / W_NEW - PADDING;
		int ww = hw % W_NEW - PADDING;

		for (int i = 0; i < 16; i++)
		{
			org_data[i] = 0.0f;
		}

		bool needFetching = (ww >=0) && (ww < W) && (hh >= 0) && (hh < H);
		if (needFetching == true) {
			int base = n * C * H * W + cc * H * W + hh * W + ww;
			for (int i = 0; i < RW_PER_THREAD; i++)
			{
				org_data[i] = bufA[base + i * H * W];
			}
		}
		int base = n * C * H_NEW * W_NEW + cc * H_NEW * W_NEW + hw;
		for (int i = 0; i < RW_PER_THREAD; i++)
		{
			bufB[base + i * H_NEW * W_NEW] = org_data[i];
		}
	}
}


using namespace std;

int main() {

	float* hostA;
	float* hostB;
	
	float* deviceA;
	float* deviceB;

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

	hostA = (float*)malloc(NUM * sizeof(float));
	hostB = (float*)malloc(NUM * sizeof(float));

	float* p;

	p = hostA;
	for (int i = 0; i < NUM; i++)
	{
		p[i] = float(i);
	}

	HIP_ASSERT(hipMalloc((void**)& deviceA, NUM * sizeof(float)));
	HIP_ASSERT(hipMalloc((void**)& deviceB, NUM * sizeof(float)));
	HIP_ASSERT(hipMemcpy(deviceA, hostA, NUM * sizeof(float), hipMemcpyHostToDevice));


	hipLaunchKernel(test_kernel,
		dim3(1, 1, 1),
		dim3(64, 1, 1),
		0, 0,
		deviceA, deviceB);
	{
		hipEventRecord(start, NULL);
		hipLaunchKernel(test_kernel,
			dim3(1, C/RW_PER_THREAD, N),
			dim3(256, 1, 1),
			0, 0,
			deviceA, deviceB);

		hipEventRecord(stop, NULL);
		hipEventSynchronize(stop);

		hipEventElapsedTime(&eventMs, start, stop);

		printf("Read/Write [%d] Channels per thread:  elapsed time:%f\n", (int)RW_PER_THREAD, eventMs);

		double bandwidth = (double)N * (double)C * (double)H_NEW * (double)W_NEW * sizeof(float) * 2 /1024/1024/1024 / (eventMs / 1000.0);

		printf("Read/Write [%d] Channels per thread:  ==> Estimated Bandwidth %d  GB/s\n", (int)RW_PER_THREAD, (int)bandwidth);

	}
	HIP_ASSERT(hipFree(deviceA));
	HIP_ASSERT(hipFree(deviceB));

	free(hostA);
	free(hostB);

	return errors;
}
