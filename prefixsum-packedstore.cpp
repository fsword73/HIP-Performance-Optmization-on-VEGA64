#include <assert.h>
#include <stdio.h>
#include <algorithm>
#include <stdlib.h>
#include <iostream>
#include "hip/hip_runtime.h"
#include <math.h>



#define HIP_ASSERT(x) (assert((x)==hipSuccess))

//Algorithm£º 
//Step 0:  Prefix sum  every 256 mask into 1 GROUP_IDX
//step 1:  Prefix sum  every 256 group idx 
//step 2:  prefix sum  every 256x256 group idx
//step 3:  prefix sum  every 256x256 group idx
//step 4:  packed store by


__global__ void mask_prefixsum(hipLaunchParm lp, const unsigned char* mask, unsigned int* group_offset, int total_length)
{
	int localId = hipThreadIdx_x;
	int localSize = hipBlockDim_x;
	int globalIdx = hipBlockIdx_x;
	__shared__ unsigned int prefix_sum[1024];

	globalIdx = globalIdx * localSize + localId;

	prefix_sum[localId] = 0;
	if (globalIdx < total_length)
	{
		if (mask[globalIdx] > 0)
		{
			prefix_sum[localId] = 1;
		}
	}


	//Do Prefix Sum;
	//https://en.wikipedia.org/wiki/Prefix_sum
	//Circuit representation of a highly parallel 16-input parallel prefix sum
	for (int ll = 1; ll <= localSize / 2; ll = ll << 1)
	{
		__syncthreads();
		if (localId >= ll)
		{
			int pre_val = prefix_sum[localId - ll];
			int cur_val = prefix_sum[localId];
			cur_val += pre_val;
			__syncthreads();
			prefix_sum[localId] = cur_val;
		}
	}

	__syncthreads();

	//Store max value into hipBlockIdx_X 
	if (0 == hipThreadIdx_x)
	{
		//Get Max from Last Valid Shared Memory
		unsigned int max_value = prefix_sum[hipBlockDim_x - 1];
		group_offset[hipBlockIdx_x] = max_value;
	}
}

__global__ void group_prefixsum_failed_function_single_barrier(hipLaunchParm lp, unsigned int* group_offset, int total_length)
{
	int localId = hipThreadIdx_x;
	int localSize = hipBlockDim_x;
	int globalIdx = hipBlockIdx_x;
	__shared__ unsigned int prefix_sum[1024];

	globalIdx = globalIdx * localSize + localId;


	prefix_sum[localId] = 0;
	if (globalIdx < total_length)
	{
		prefix_sum[localId] = group_offset[globalIdx];
	}


	//Do Prefix Sum;
	//Circuit representation of a highly parallel 16-input parallel prefix sum
	for (int ll = 1; ll <= localSize / 2; ll = ll << 1)
	{
		__syncthreads();
		if (localId >= ll) {
			int pre_val = prefix_sum[localId - ll];
			int cur_val = prefix_sum[localId];
			cur_val += pre_val;
			//__syncthreads();
			prefix_sum[localId] = cur_val;
		}
	}

	__syncthreads();

	//Store max value into hipBlockIdx_X 
	if (globalIdx < total_length)
		group_offset[globalIdx] = prefix_sum[localId];
}

__global__ void group_prefixsum_Pass1(hipLaunchParm lp, unsigned int* group_offset, unsigned int total_length)
{
	int localId = hipThreadIdx_x;
	int localSize = hipBlockDim_x;
	int globalIdx = hipBlockIdx_x;
	__shared__ unsigned int prefix_sum[1024];

	globalIdx = globalIdx * localSize + localId;


	prefix_sum[localId] = 0;
	if (globalIdx < total_length)
	{
			prefix_sum[localId] = group_offset[globalIdx];
	}


	//Do Prefix Sum;
	//Circuit representation of a highly parallel 16-input parallel prefix sum
	for (int ll = 1; ll <= localSize / 2; ll = ll << 1)
	{
		__syncthreads();
		if (localId >= ll)		{
			int pre_val = prefix_sum[localId - ll];
			int cur_val = prefix_sum[localId];
			cur_val += pre_val;
			__syncthreads();
			prefix_sum[localId] = cur_val;
		}
	}

	__syncthreads();

	//Store max value into hipBlockIdx_X 
	if(globalIdx < total_length)
		group_offset[globalIdx] = prefix_sum[localId];
}


__global__ void group_prefixsum_Pass2(hipLaunchParm lp, unsigned int* group_offset, unsigned int* group_offset2, unsigned int offset, unsigned int total_length)
{
	unsigned int globalIdx = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
	unsigned int localId = hipThreadIdx_x;
	__shared__ unsigned int prefix_sum[1024];

	unsigned int big_group_size = offset * hipBlockDim_x;
	unsigned int big_group_base = (hipBlockIdx_x * hipBlockDim_x) & (~(big_group_size - 1));

	//inside its own offset*hipBlockDim_x, so must from offset gtroup 1,
	//Otherwise it will read 1 more data from previous group's last data
	
	unsigned int to_prefixsum_idx = big_group_base + (hipThreadIdx_x+1) * offset -1;


	if (globalIdx < offset)
	{
		group_offset2[globalIdx] = group_offset[globalIdx];
		return;
	}
		

	unsigned int to_prefixsum_val = 0;


	//from second offset * hipBlockDim_x:  can not read 1 
	if ((to_prefixsum_idx < (hipBlockIdx_x * hipBlockDim_x)) )
	
		to_prefixsum_val = group_offset[to_prefixsum_idx];

	prefix_sum[hipThreadIdx_x] = to_prefixsum_val;

	unsigned int cur_val = 0;
	if(globalIdx < total_length)
		cur_val = group_offset[globalIdx];

	__syncthreads();
	//DO PrefixSUm of (max hipBlockDim_x ) 
	for (int ll = 1; ll <= hipBlockDim_x / 2; ll = ll << 1)
	{
		__syncthreads();
		if (localId >= ll) {
			int pre_val = prefix_sum[localId - ll];
			int cur_val = prefix_sum[localId];
			cur_val += pre_val;
			__syncthreads();
			prefix_sum[localId] = cur_val;
		}
	}


	__syncthreads();

	if (globalIdx >= total_length)
		return;
	 
	unsigned int max_val = prefix_sum[hipBlockDim_x - 1];


	group_offset2[globalIdx] = max_val + cur_val;

}


__global__ void packed_store_prefixsum(hipLaunchParm lp, const unsigned int* in, const unsigned char* mask, unsigned int* out, unsigned int* group_offset, unsigned int total_length)
{
	int localId = hipThreadIdx_x;
	int localSize = hipBlockDim_x;
	int globalIdx = hipBlockIdx_x * hipBlockDim_x + hipThreadIdx_x;
	__shared__ unsigned int prefix_sum[1024];  //



	if (globalIdx >= total_length)
		return;

	int group_prefixsum = 0;

	//1st workgroup offset begins from 0
	if (hipBlockIdx_x > 0)
	{
		group_prefixsum = group_offset[hipBlockIdx_x - 1];
	}

	unsigned int mask_val = 0;
	mask_val = mask[globalIdx];

	if (mask_val > 0)
		mask_val = 1;

	prefix_sum[localId] = mask_val;

	//Do Prefix Sum locally;
	for (int ll = 1; ll <= localSize / 2; ll = ll << 1)
	{
		__syncthreads();
		if (localId >= ll) {
			int pre_val = prefix_sum[localId - ll];
			int cur_val = prefix_sum[localId];
			cur_val += pre_val;
			__syncthreads();
			prefix_sum[localId] = cur_val;
		}
	}
	__syncthreads();
	if (mask_val > 0)
	{
		//first thread with Mask>1,  the prefix_sum result for 1st thread = 1, so must -1s,  
		out[group_prefixsum + prefix_sum[localId] - 1] = in[globalIdx];	
	}
}

using namespace std;

void PackedStore(const unsigned  int* in, const unsigned char* mask, unsigned  int* out, unsigned int* groupoffset, unsigned int* groupoffset2, unsigned int inlength, unsigned int& outlength, unsigned int verify = 0) {
	unsigned int localThreads = 256;
	unsigned int gloal_blocks = (inlength + localThreads - 1) / localThreads;


	//DO Prefix for every 256 threads
	hipLaunchKernel(mask_prefixsum,
		dim3(gloal_blocks),
		dim3(localThreads),
		0, 0,
		mask, groupoffset, inlength);


#if 0
	if (verify > 0) {


		unsigned int packedLen = 0;
		unsigned int* tempBuf = (unsigned int*)malloc(sizeof(unsigned int) * (gloal_blocks));
		HIP_ASSERT(hipMemcpy(tempBuf, groupoffset, sizeof(unsigned int) * (gloal_blocks), hipMemcpyDeviceToHost));
		hipStreamSynchronize(0);


		for (int i = 0; i < gloal_blocks; i++)
		{
			if (128 != tempBuf[i])
			{
				if (i != (gloal_blocks - 1))
					printf("must == 128 , [%d]th %9d %9d\n", i, tempBuf[i], tempBuf[i - 1]);
			}

		}
		printf("passed for mask_prefixsum\n");
		free(tempBuf);
	}
#endif
	//Prefix Sum for GroupOffset
	unsigned int offset = 1;
	unsigned int group_blocks = (gloal_blocks + localThreads - 1) / localThreads;

	for (offset = 1; offset < gloal_blocks; offset = offset * localThreads)
	{

		if (offset == 1) {
			hipLaunchKernel(group_prefixsum_Pass1,
				dim3(group_blocks),
				dim3(localThreads),
				0, 0,
				groupoffset, gloal_blocks);
	

		}
		else if (offset == 256){

			hipLaunchKernel(group_prefixsum_Pass2,
				dim3(group_blocks),
				dim3(localThreads),
				0, 0,
				groupoffset, groupoffset2, offset, gloal_blocks);


		}
		else 
		{
			hipLaunchKernel(group_prefixsum_Pass2,
				dim3(group_blocks),
				dim3(localThreads),
				0, 0,
				groupoffset2, groupoffset, offset, gloal_blocks);
		}	

	}


#if 0
	if (verify > 0) {
		if ((gloal_blocks > 256) && (gloal_blocks <= 256 * 256)) 
		{
			HIP_ASSERT(hipMemcpyDtoD(groupoffset, groupoffset2, sizeof(unsigned int) * (gloal_blocks)));
		}

		unsigned int packedLen = 0;
		unsigned int* tempBuf = (unsigned int*)malloc(sizeof(unsigned int) * (gloal_blocks));
		HIP_ASSERT(hipMemcpy(tempBuf, groupoffset, sizeof(unsigned int) * (gloal_blocks), hipMemcpyDeviceToHost));
		hipStreamSynchronize(0);


		for (int i = 0; i < gloal_blocks; i++)
		{
			if ((128 * (i+1))  != tempBuf[i])
			{
				if(i != (gloal_blocks-1))
					printf(" not equal 128x == , [%d]th %9d \n", i, tempBuf[i]);
			}

		}
		free(tempBuf);
	}

#endif 

	if ((gloal_blocks > 256) && (gloal_blocks <= 256 * 256))
	{
		HIP_ASSERT(hipMemcpy(&outlength, groupoffset2 + (gloal_blocks - 1), sizeof(unsigned int) * 1, hipMemcpyDeviceToHost));

		//Finally output
		hipLaunchKernel(packed_store_prefixsum,
			dim3(gloal_blocks),
			dim3(localThreads),
			0, 0,
			in, mask, out, groupoffset2, inlength);

	}
	else
	{
		HIP_ASSERT(hipMemcpy(&outlength, groupoffset + (gloal_blocks - 1), sizeof(unsigned int) * 1, hipMemcpyDeviceToHost));

		//Finally output
		hipLaunchKernel(packed_store_prefixsum,
			dim3(gloal_blocks),
			dim3(localThreads),
			0, 0,
			in, mask, out, groupoffset, inlength);
	}
}

void test(unsigned int inlength)
{
	unsigned int  A_NUM = inlength;
	unsigned int  B_NUM = inlength;
	unsigned int  C_NUM = inlength;
	unsigned int  D_NUM = ((inlength + 256 - 1) / 256);


	unsigned int* hostA;
	unsigned char* hostB;
	unsigned int* hostC;
	unsigned int* hostD;

	unsigned int* deviceA;
	unsigned char* deviceB;
	unsigned int* deviceC;
	unsigned int* deviceD;
	unsigned int* deviceD2;

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

	hostA = (unsigned int*)malloc(A_NUM * sizeof(unsigned int));
	hostB = (unsigned char*)malloc(B_NUM * sizeof(unsigned char));
	hostC = (unsigned int*)malloc(C_NUM * sizeof(unsigned int));
	hostD = (unsigned int*)malloc(D_NUM * sizeof(unsigned int));

	for (unsigned int i = 0; i < inlength; i = i + 1) {
		hostA[i] = i;
		hostB[i] = (i) & (0x1);   //Magic Mask
		//printf("%d ", hostB[i]);
		hostC[i] = 0;
	}
	//printf("\n");


	cout << "host allocated \n";
	HIP_ASSERT(hipMalloc((void**)& deviceA, A_NUM * sizeof(unsigned int)));
	HIP_ASSERT(hipMalloc((void**)& deviceB, B_NUM * sizeof(unsigned char)));
	HIP_ASSERT(hipMalloc((void**)& deviceC, C_NUM * sizeof(unsigned int)));
	HIP_ASSERT(hipMalloc((void**)& deviceD, D_NUM * sizeof(unsigned int)));
	HIP_ASSERT(hipMalloc((void**)& deviceD2, D_NUM * sizeof(unsigned int)));


	cout << "device allocated \n";
	HIP_ASSERT(hipMemcpy(deviceA, hostA, A_NUM * sizeof(unsigned int), hipMemcpyHostToDevice));
	HIP_ASSERT(hipMemcpy(deviceB, hostB, B_NUM * sizeof(unsigned char), hipMemcpyHostToDevice));
	HIP_ASSERT(hipMemcpy(deviceC, hostC, C_NUM * sizeof(unsigned int), hipMemcpyHostToDevice));

	cout << "Host to Device Copied\n";

	{

		unsigned int outLength = 0;
		hipStreamSynchronize(0);
		unsigned verify = 0;
		PackedStore(deviceA, deviceB, deviceC, deviceD, deviceD2, inlength, outLength, verify);
		hipStreamSynchronize(0);

		printf("inLen: %d, outLength %d\n", inlength, outLength);
		if (verify > 0) {
			unsigned int  targetLen = (inlength / 2) + (inlength & 0x1);
			unsigned int  targetVal = (inlength - 1) & 0xFFFFFFFF;
			unsigned int  packedVal = 0;
			HIP_ASSERT(hipMemcpy(hostC, deviceC, sizeof(unsigned int) * (targetLen), hipMemcpyDeviceToHost));
			hipStreamSynchronize(0);
			packedVal = hostC[targetLen - 1];
			if (targetVal == packedVal)
			{
				printf("--->Passed!  target [%d]  == packed [%d] \n", targetVal, packedVal);
			}
			else
			{
				printf("--->Failed!  target [%d]  == packed [%d] \n", targetVal, packedVal);

			}

		}

		hipStreamSynchronize(0);

		hipEventRecord(start, NULL);

		int  iterations = 10;
		for (int i = 0; i < iterations; i++)
		{
			PackedStore(deviceA, deviceB, deviceC, deviceD, deviceD2, inlength, outLength, 0);
			if (0) {
				unsigned int* tmpBuffer = (unsigned int*)malloc(outLength * sizeof(unsigned int));
				HIP_ASSERT(hipMemcpy(tmpBuffer, deviceC , outLength * sizeof(unsigned int), hipMemcpyDeviceToHost));
				free(tmpBuffer);
			}
		}

		hipEventRecord(stop, NULL);
		hipEventSynchronize(stop);
		hipEventElapsedTime(&eventMs, start, stop);

		float bandwidth = float(inlength) * float( sizeof(unsigned int) * 2 + sizeof(unsigned char) * 2) / (float)1024 / (float)1024 / (float)1024;
		bandwidth = bandwidth / (eventMs / iterations / 1000.0);
		printf("Inlen:[%d] costs %f millseconds ,  Bandwidth = %f GB/s\n", inlength, eventMs / iterations, bandwidth);

	}
	HIP_ASSERT(hipMemcpy(hostC, deviceC, sizeof(int) * C_NUM, hipMemcpyDeviceToHost));

	//CPU Verifying here
	HIP_ASSERT(hipFree(deviceA));
	HIP_ASSERT(hipFree(deviceB));
	HIP_ASSERT(hipFree(deviceC));
	HIP_ASSERT(hipFree(deviceD));
	HIP_ASSERT(hipFree(deviceD2));


	free(hostA);
	free(hostB);
	free(hostC);
	free(hostD);

	//return errors;
}


int main(int argc, char** argv)
{
	if (argc != 2)
	{
		printf("it is to demo packed store:\n");
		printf("     <length>   total length to do pack store \n");
		printf("----Packed Store---	\n");
		printf("void func(uint32_t inlen, const uint32_t *in, const uint8_t *mask, uint32_t *out) { \n");
		printf("     uint32_t oidx = 0; \n");
		printf("     for (uint32_t_t i = 0; i < inlen; i++) { \n");
		printf("     if (mask[i]) out[oidx++] = in[i]; \n");
		printf("}\n");
		return 0;
	}
	unsigned int packedLength = 0;

	packedLength = atoi(argv[1]);

	//packedLength = 256 * 256 * 255;
	test(packedLength);

	return 0;
}

