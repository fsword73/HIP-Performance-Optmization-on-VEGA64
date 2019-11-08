#include <assert.h>
#include <stdio.h>
#include <algorithm>
#include <stdlib.h>
#include <iostream>
#include "hip/hip_runtime.h"
#include <math.h>


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

	__syncthreads();

	
	//Scatter 256xH*W to different thread
	int tile_c_offset = c & (~0xff); 
	int dest_base = k * CC * HH * WW + tile_c_offset * HH * WW + hipThreadIdx_x;
	int shared_Idx2 = hipThreadIdx_x;
	for (int i = 0; i < HH; i++)
		for (int j = 0; j < WW; j++)
		{
			dest[dest_base] = shared_data[shared_Idx2];
			shared_Idx2 += 256;
			dest_base += 256;
		}
}

using namespace std;

#define KC_VAL  4096

void test(unsigned int k= KC_VAL, int c= KC_VAL, int h=3, int w=3)
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

	//hostA = (float*)malloc(A_NUM * sizeof(float));
	hipHostMalloc((void**)& hostA, A_NUM * sizeof(float));
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
		int gloal_blocks = (inlength + localThreads - 1) / localThreads;

		hipLaunchKernel(convert_nhwc_to_nchw<KC_VAL, KC_VAL, 3, 3>,
			dim3(gloal_blocks),
			dim3(localThreads),
			0, 0,
			hostA, deviceB);

		hipEventRecord(start, NULL);

		int  iterations = 10;

		for (int i = 0; i < iterations; i++)
		{
			hipLaunchKernel(convert_nhwc_to_nchw<KC_VAL, KC_VAL, 3, 3>,
				dim3(gloal_blocks),
				dim3(localThreads),
				0, 0,
				hostA, deviceB);
		}

		hipEventRecord(stop, NULL);
		hipEventSynchronize(stop);
		hipEventElapsedTime(&eventMs, start, stop);

		float bandwidth = float(inlength) * 1 / 1e9;
		bandwidth = bandwidth / (eventMs / iterations / 1000.0);
		printf("Shuffle_simple : host A, Inlen:[%d], costs %f millseconds ,  Bandwidth = %f Giga Data/s\n", inlength, eventMs / iterations, bandwidth);
	}

	{
		int localThreads = 256;
		int gloal_blocks = (inlength + localThreads -1) / localThreads;

		hipLaunchKernel(convert_nhwc_to_nchw<KC_VAL, KC_VAL,3,3>,
			dim3(gloal_blocks),
			dim3(localThreads),
			0, 0,
			deviceA, deviceB);

		hipEventRecord(start, NULL);

		int  iterations = 10;
		
		for (int i = 0; i < iterations; i++)
		{
			hipLaunchKernel(convert_nhwc_to_nchw<KC_VAL, KC_VAL, 3, 3>,
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

		hipLaunchKernel(convert_nhwc_to_nchw_optimized<KC_VAL, KC_VAL, 3, 3>,
			dim3(gloal_blocks),
			dim3(localThreads),
			0, 0,
			deviceA, deviceB);

		hipEventRecord(start, NULL);

		int  iterations = 10;

		for (int i = 0; i < iterations; i++)
		{
			hipLaunchKernel(convert_nhwc_to_nchw_optimized<KC_VAL, KC_VAL, 3, 3>,
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


	//free(hostA);
	hipFree(hostA);
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

