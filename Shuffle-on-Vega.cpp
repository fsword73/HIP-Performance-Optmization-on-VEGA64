#include <assert.h>
#include <stdio.h>
#include <algorithm>
#include <stdlib.h>
#include <iostream>
#include "hip/hip_runtime.h"
#include <math.h>

/*
	This test is aimed to test the performance of Atomic function vs redution 
	especially for batched GEMM and Convoloution Backweights when Channels are very small but H*W is very huge

•	Method 1:  Per-Thread VGPR v_add_f32 function only which is the fastest one. We don’t write test for it . At least 32x32 macro-tile can be used for SGEMM and convolution .  Matrix K can be Split 16x at most
•	Method 2:  Per-Wave LDS Reduction.   It is very fast. It has  lots of test.  We don’t care.    At least 32x32 macro-tile can be used for SGEMM and convolution.   Matrix K can be Split 16x at most
o	Workgroup size 256
o	Real  Macro-tile size 32x32 , faked macro tile size 128x128
o	Split K = 16
o	Every thread does 4x4 result C.  Every thread has  4x K
o	Every 4 Threads have 4x4 K =  Split K = 16
o	It still has 128x128 faked macro-tile size

•	Method 3:  Using Atomic_add_int to simulate the atomic_float function in MI-100
•	Method4:   Using Hip function atomic_add_float in ROCm HIP.   Actually  it uses while loop and global_load_buffer,  and atomic_cmpexchange
inline void AtomicAdd(volatile __global float *source, const float operand) {
	union {
		unsigned int intVal;
		float floatVal;
	} newVal;
	union {
		unsigned int intVal;
		float floatVal;
	} prevVal;
	do {
		prevVal.floatVal = *source;
		newVal.floatVal = prevVal.floatVal + operand;
	} while (atomic_cmpxchg((volatile __global unsigned int *)source,
							 prevVal.intVal, newVal.intVal)
							 != prevVal.intVal);
o	}
•	Method 5: Assuming that No hardware atomic function supported.
Use 2 Passes for BatchGEMM or convolution. For example,  1st Kernel outputs split-K result into 64x temp buffer.  2nd kernel loads 64 Matrix C and sums up into final buffer.
	   for (int i = 0; i < Redcutions; i++)
	   {
			  result += srcdata[globalIdx + total_length];
	   }

	   dstdata[globalIdx] = result;


*/

#define HIP_ASSERT(x) (assert((x)==hipSuccess))


template<int KK, int CC, int HH, int WW>
__global__ void  convert_nhwc_to_nchw(hipLaunchParm lp, float* source, float* dest)
{
	int global_idx = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

	int k, c, h, w;

	
	w = global_idx % WW;
	int temp = global_idx / WW;
	h = temp % HH;
	temp = temp /HH;

	c = temp % CC;
	temp = temp / CC;
	k = temp;

	//NHWC
	int soruce_idx = k *  HH * WW *CC + h * WW * CC + w *CC + c;
	dest[global_idx] = source[soruce_idx];


}

template<int KK, int CC, int HH, int WW>
__global__ void  convert_nhwc_to_nchw_optimized(hipLaunchParm lp, float* source, float* dest)
{
	//Padding 1
	const int stride = 257;
	__shared__ float shared_data[stride * 9];

	//global_idx = KK * CC;
	int global_idx = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;

	int k, c;

	c = global_idx % CC;
	k = global_idx / CC;

	
	int source_base = k * HH * WW * CC + c;
	for(int i=0; i < HH; i++)
		for (int j = 0; j < WW; j++)
		{
			int h = i;
			int w = j;
			int soruce_idx = source_base + h * WW * CC + w * CC ;

			int shared_idx = (c&0xff) *HH *WW + h * WW + w;
			shared_data[shared_idx] = source[soruce_idx];
		}


	//Scatter 256xH*W to different thread
	int tile_c_offset = c & (~0xff); 
	int dest_base = k * CC * HH * WW + tile_c_offset * HH * WW + hipThreadIdx_x;
	int shared_Idx2 = hipThreadIdx_x;
	for (int i = 0; i < HH; i++)
		for (int j = 0; j < WW; j++)
		{
			shared_Idx2 += 256 ;
			dest_base += 256;
			dest[dest_base] = shared_data[shared_Idx2];
		}
}

using namespace std;

void test(unsigned int k=512, int c=512, int h=3, int w=3)
{
	int inlength = k * c * h * w;
	unsigned int  A_NUM = inlength;
	unsigned int  B_NUM = inlength;
	unsigned int  C_NUM = inlength;


	float* hostA;
	float* hostB;
	float* hostC;


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

	int errors;

	hostA = (float*)malloc(A_NUM * sizeof(float));
	hostB = (float*)malloc(B_NUM * sizeof(float));
	hostC = (float*)malloc(C_NUM * sizeof(float));

	for (unsigned int i = 0; i < inlength; i = i + 1) {
		hostA[i] = i;
		hostB[i] = i;
		hostC[i] = i;
	}

	float* deviceA;
	float* deviceB;
	float* deviceC;

	cout << "host allocated \n";
	HIP_ASSERT(hipMalloc((void**)& deviceA, A_NUM * sizeof(float)));
	HIP_ASSERT(hipMalloc((void**)& deviceB, B_NUM * sizeof(float)));
	HIP_ASSERT(hipMalloc((void**)& deviceC, C_NUM * sizeof(float)));


	cout << "device allocated \n";
	HIP_ASSERT(hipMemcpy(deviceA, hostA, A_NUM * sizeof(float), hipMemcpyHostToDevice));
	HIP_ASSERT(hipMemcpy(deviceB, hostB, B_NUM * sizeof(float), hipMemcpyHostToDevice));
	HIP_ASSERT(hipMemcpy(deviceC, hostC, C_NUM * sizeof(float), hipMemcpyHostToDevice));

	cout << "Host to Device Copied\n";

	{
		int localThreads = 256;
		int gloal_blocks = (inlength + localThreads -1) / localThreads;

		hipLaunchKernel(convert_nhwc_to_nchw<512,512,3,3>,
			dim3(gloal_blocks),
			dim3(localThreads),
			0, 0,
			deviceA, deviceB);

		hipEventRecord(start, NULL);

		int  iterations = 10;
		
		for (int i = 0; i < iterations; i++)
		{
			hipLaunchKernel(convert_nhwc_to_nchw<512, 512, 3, 3>,
				dim3(gloal_blocks),
				dim3(localThreads),
				0, 0,
				deviceA, deviceB);
		}

		hipEventRecord(stop, NULL);
		hipEventSynchronize(stop);
		hipEventElapsedTime(&eventMs, start, stop);

		float bandwidth = float(inlength) * 1 / 1e9  ;
		bandwidth = bandwidth / (eventMs / iterations / 1000.0);
		printf("Shuffle_simple, Inlen:[%d], costs %f millseconds ,  Bandwidth = %f Giga Data/s\n", inlength, eventMs / iterations, bandwidth);
	}

	{
		int localThreads = 256;
		
		//9*256 once
		int gloal_blocks = (inlength/9 + localThreads - 1) / localThreads;

		hipLaunchKernel(convert_nhwc_to_nchw_optimized<512, 512, 3, 3>,
			dim3(gloal_blocks),
			dim3(localThreads),
			0, 0,
			deviceA, deviceB);

		hipEventRecord(start, NULL);

		int  iterations = 10;

		for (int i = 0; i < iterations; i++)
		{
			hipLaunchKernel(convert_nhwc_to_nchw_optimized<512, 512, 3, 3>,
				dim3(gloal_blocks),
				dim3(localThreads),
				0, 0,
				deviceA, deviceB);
		}

		hipEventRecord(stop, NULL);
		hipEventSynchronize(stop);
		hipEventElapsedTime(&eventMs, start, stop);

		float bandwidth = float(inlength) * 1 / 1e9;
		bandwidth = bandwidth / (eventMs / iterations / 1000.0);
		printf("Shuffle_optimized, Inlen:[%d], costs %f millseconds ,  Bandwidth = %f Giga Data/s\n", inlength, eventMs / iterations, bandwidth);
	}



	//CPU Verifying here
	HIP_ASSERT(hipFree(deviceA));
	HIP_ASSERT(hipFree(deviceB));
	HIP_ASSERT(hipFree(deviceC));


	free(hostA);
	free(hostB);
	free(hostC);

	//return errors;
}


int main(int argc, char** argv)
{
	if (argc != 2)
	{	
		//printf(" please input length\n\n");
		return 0;
	}
	unsigned int packedLength = 0;

	//packedLength = atoi(argv[1]);

	//packedLength = 256 * 256 * 255;
	test();

	return 0;
}

