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
#include <math.h>



#define HIP_ASSERT(x) (assert((x)==hipSuccess))


#define N   128
#define C   1024
#define H   28
#define W   28

#define  NUM    ( N * C * H * W )
__global__ void
test_kernel(hipLaunchParm lp,
	float* __restrict__ bufA, float* __restrict__ bufB,  int n, int chw, float gamma, float bata )
{
	   
	int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;

	float  fmean = 0;
	float fstd = 0;

	for (int i = 0; i < n; i++)
	{
		fmean += bufA[x + i * chw];
	}
	fmean = fmean / n;

	for (int i = 0; i < n; i++) 
	{
		float v = bufA[x + i * chw] - fmean;
		fstd += v * v;
	}

	fstd = fstd / n;

	float epsilon = 1e-6;

	fstd = rsqrtf(fstd + epsilon);
	
	float result = 0;

	for (int i = 0; i < n; i++)
	{
		float v = bufA[x + i * chw];
		result = gamma * (v - fmean) * fstd + bata;

		bufB[x + i * chw] = result;
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
		p[i] = float(sinf(i));
	}

	HIP_ASSERT(hipMalloc((void**)& deviceA, NUM * sizeof(float)));
	HIP_ASSERT(hipMalloc((void**)& deviceB, NUM * sizeof(float)));
	HIP_ASSERT(hipMemcpy(deviceA, hostA, NUM * sizeof(float), hipMemcpyHostToDevice));


	hipLaunchKernel(test_kernel,
		dim3(1, 1, 1),
		dim3(1, 1, 1),
		0, 0,
		deviceA, deviceB, 1, 1, 1.0, 0.0);

	{
		hipEventRecord(start, NULL);
		hipLaunchKernel(test_kernel,
			dim3(C*H*W / 256, 1,1),
			dim3(256, 1, 1),
			0, 0,
			deviceA, deviceB, N, C*H*W, 1.0f, 1.0f);

		hipEventRecord(stop, NULL);
		hipEventSynchronize(stop);

		hipEventElapsedTime(&eventMs, start, stop);

		printf("elapsed time:%f\n",  eventMs);

		double bandwidth = (double)N * (double)C * (double)H * (double)W  / (eventMs / 1000.0)/1000/1000/1000;

		printf("Estimated Bandwidth %d  GPixels/s\n", (int)bandwidth);

	}
	HIP_ASSERT(hipFree(deviceA));
	HIP_ASSERT(hipFree(deviceB));

	free(hostA);
	free(hostB);

	return errors;
}
