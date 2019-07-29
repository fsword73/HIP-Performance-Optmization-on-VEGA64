/*
please note that the series of optmiztion technology is not in official document.

All the tests are based on AMD MI25 radeon instict and AMD ROCm.
*/

// 目标产生NCHW格式的NHW
// 存在一个问题如何优化， 如果HW < (16*16)


#include <assert.h>
#include <stdio.h>
#include <algorithm>
#include <stdlib.h>
#include<iostream>
#include "hip/hip_runtime.h"



#define HIP_ASSERT(x) (assert((x)==hipSuccess))

#define N        1024 
#define C        64
#define H        56
#define W        56  

#define NUM  (N * H * W) 

#define THREADS_PER_BLOCK_X  256
#define THREADS_PER_BLOCK_Y  1
#define THREADS_PER_BLOCK_Z  1

#define BLOCKS_X    (( H * W + THREADS_PER_BLOCK_X-1) / THREADS_PER_BLOCK_X)



//local block size,  (256, 1)
//total threads (H * W, 1)
//


__global__ void
test_kernel(hipLaunchParm lp,
	int* __restrict__ buf, int h, int w, int c)
{

	int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
	int n = hipBlockIdx_y;

	if (x < (h * w))
	{
		int nchw_offset = x + n * c * h * w;
		int nhw_offset = x + n * h * w;
		buf[nhw_offset] = nchw_offset;
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



	HIP_ASSERT(hipMalloc((void**)& deviceA, NUM * sizeof(int)));
	HIP_ASSERT(hipMemcpy(deviceA, hostA, NUM * sizeof(int), hipMemcpyHostToDevice));

	hipLaunchKernel(test_kernel,
		dim3(1, 1, 1),
		dim3(THREADS_PER_BLOCK_X, 1, 1),
		0, 0,
		deviceA, 1, 1, 1);


	hipEventRecord(start, NULL);
	for (int i = 0; i < 100; i = i + 1) {
		hipLaunchKernel(test_kernel,
			dim3(BLOCKS_X, N, 1),
			dim3(THREADS_PER_BLOCK_X, 1, 1),
			0, 0,
			deviceA, H, W, C);
	}
	hipEventRecord(stop, NULL);
	hipEventSynchronize(stop);

	hipEventElapsedTime(&eventMs, start, stop);

	double bandwidth = double(N) * double(H) * double(W) * sizeof(int) /  1024 / 1024 / 1024 / (eventMs / 100 / 1000);
	printf("N*H*W=[%d,%d,%d], kernel_time (hipEventElapsedTime) =%6.3f microseconds, %3f GB/s\n", N, H, W, eventMs /100 * 1000,bandwidth);

	HIP_ASSERT(hipFree(deviceA));

	free(hostA);

	return errors;
}
