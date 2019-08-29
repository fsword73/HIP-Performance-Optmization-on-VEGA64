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

__global__ void  test_atomic_int(hipLaunchParm lp, unsigned int* data, int total_length)
{

	int globalIdx = hipBlockIdx_y * hipBlockDim_x + hipThreadIdx_x;
	if (globalIdx >= total_length)
		return;

	atomicAdd(data + globalIdx, hipBlockIdx_x);
}

__global__ void  test_atomic_float(hipLaunchParm lp,  float* data, int total_length)
{
	int globalIdx = hipBlockIdx_y * hipBlockDim_x + hipThreadIdx_x;
	if (globalIdx >= total_length)
		return;

	atomicAdd(data + globalIdx, (float)hipBlockIdx_x);
}

template<int Redcutions>
__global__ void test_direct_reduction(hipLaunchParm lp, const float* srcdata, float* dstdata,  int total_length)
{
	int globalIdx = hipBlockIdx_y * hipBlockDim_x + hipThreadIdx_x;

	float result = 0;

	if (globalIdx >= total_length)
		return;

	for (int i = 0; i < Redcutions; i++)
	{
		result += srcdata[globalIdx + total_length];
	}

	dstdata[globalIdx] = result;
}

using namespace std;

template<int COUNT>
void test_direct_add(float* src, float* dest, int inlength)
{
	hipEvent_t start, stop;
	hipEventCreate(&start);
	hipEventCreate(&stop);
	float eventMs = 1.0f;


	int localThreads = 256;
	int gloal_blocks = (inlength + localThreads - 1) / localThreads;
	int  l = COUNT;

	hipLaunchKernel(test_direct_reduction<COUNT>,
		dim3(l, gloal_blocks),
		dim3(localThreads),
		0, 0,
		src, dest, inlength);

	hipEventRecord(start, NULL);

	int  iterations = 10;
	//  test_kernel;
	for (int i = 0; i < iterations; i++)
	{
		hipLaunchKernel(test_direct_reduction<COUNT>,
			dim3(l, gloal_blocks),
			dim3(localThreads),
			0, 0,
			src, dest, inlength);
	}

	hipEventRecord(stop, NULL);
	hipEventSynchronize(stop);
	hipEventElapsedTime(&eventMs, start, stop);

	float bandwidth = float(inlength) * l / 1e9;
	bandwidth = bandwidth / (eventMs / iterations / 1000.0);
	printf("Float_direct_reduction, Inlen:[%d], atomic==%d, costs %f millseconds ,  Bandwidth = %f Giga Float_direct_reduction/s\n", inlength, l, eventMs / iterations, bandwidth);


}

void test(unsigned int inlength)
{
	unsigned int  A_NUM = inlength;
	unsigned int  B_NUM = inlength;
	unsigned int  C_NUM = inlength*64;


	unsigned int* hostA;
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

	hostA = (unsigned int*)malloc(A_NUM * sizeof(unsigned int));
	hostB = (float*)malloc(B_NUM * sizeof(float));
	hostC = (float*)malloc(C_NUM * sizeof(float));

	for (unsigned int i = 0; i < inlength; i = i + 1) {
		hostA[i] = i;
		hostB[i] = i;
		hostC[i] = i;

		for(int j=1; j <64; j++ )
			hostC[i+j* inlength] = 1;
	}

	unsigned int* deviceA;
	float* deviceB;
	float* deviceC;

	cout << "host allocated \n";
	HIP_ASSERT(hipMalloc((void**)& deviceA, A_NUM * sizeof(unsigned int)));
	HIP_ASSERT(hipMalloc((void**)& deviceB, B_NUM * sizeof(float)));
	HIP_ASSERT(hipMalloc((void**)& deviceC, C_NUM * sizeof(float)));


	cout << "device allocated \n";
	HIP_ASSERT(hipMemcpy(deviceA, hostA, A_NUM * sizeof(unsigned int), hipMemcpyHostToDevice));
	HIP_ASSERT(hipMemcpy(deviceB, hostB, B_NUM * sizeof(float), hipMemcpyHostToDevice));
	HIP_ASSERT(hipMemcpy(deviceC, hostC, C_NUM * sizeof(float), hipMemcpyHostToDevice));

	cout << "Host to Device Copied\n";
	for( int l= 2; l < 64; l++)
	{
		int localThreads = 256;
		int gloal_blocks = (inlength + localThreads -1) / localThreads;

		hipLaunchKernel(test_atomic_int,
			dim3(l, gloal_blocks),
			dim3(localThreads),
			0, 0,
			deviceA, inlength);

		hipEventRecord(start, NULL);

		int  iterations = 10;
		
		for (int i = 0; i < iterations; i++)
		{
			hipLaunchKernel(test_atomic_int,
				dim3(l, gloal_blocks),
				dim3(localThreads),
				0, 0,
				deviceA, inlength);
		}

		hipEventRecord(stop, NULL);
		hipEventSynchronize(stop);
		hipEventElapsedTime(&eventMs, start, stop);

		float bandwidth = float(inlength) * l / 1e9  ;
		bandwidth = bandwidth / (eventMs / iterations / 1000.0);
		printf("test_atomic_int, Inlen:[%d], atomic==%d, costs %f millseconds ,  Bandwidth = %f Giga test_atomic_int/s\n", inlength, l, eventMs / iterations, bandwidth);
	}

	for (int l = 2; l < 64; l++)
	{
		int localThreads = 256;
		int gloal_blocks = (inlength + localThreads - 1) / localThreads;

		hipLaunchKernel(test_atomic_float,
			dim3(l, gloal_blocks),
			dim3(localThreads),
			0, 0,
			deviceB, inlength);

		hipEventRecord(start, NULL);

		int  iterations = 10;

		for (int i = 0; i < iterations; i++)
		{
			hipLaunchKernel(test_atomic_float,
				dim3(l, gloal_blocks),
				dim3(localThreads),
				0, 0,
				deviceB, inlength);
		}

		hipEventRecord(stop, NULL);
		hipEventSynchronize(stop);
		hipEventElapsedTime(&eventMs, start, stop);

		float bandwidth = float(inlength) * l / 1e9;
		bandwidth = bandwidth / (eventMs / iterations / 1000.0);
		printf("test_atomic_float, Inlen:[%d], atomic==%d, costs %f millseconds ,  Bandwidth = %f Giga atomic_add_float/s\n", inlength, l, eventMs / iterations, bandwidth);

	}

					   
	test_direct_add<2>(deviceC, deviceB , inlength);
	test_direct_add<3> (deviceC, deviceB , inlength);
	test_direct_add<4> (deviceC, deviceB , inlength);
	test_direct_add<5> (deviceC, deviceB , inlength);
	test_direct_add<6> (deviceC, deviceB , inlength);
	test_direct_add<7> (deviceC, deviceB , inlength);
	test_direct_add<8> (deviceC, deviceB , inlength);
	test_direct_add<9> (deviceC, deviceB , inlength);

	test_direct_add<10>(deviceC, deviceB , inlength);
	test_direct_add<11>(deviceC, deviceB , inlength);
	test_direct_add<12>(deviceC, deviceB , inlength);
	test_direct_add<13>(deviceC, deviceB , inlength);
	test_direct_add<14>(deviceC, deviceB , inlength);
	test_direct_add<15>(deviceC, deviceB , inlength);
	test_direct_add<16>(deviceC, deviceB , inlength);
	test_direct_add<17>(deviceC, deviceB , inlength);
	test_direct_add<18>(deviceC, deviceB , inlength);
	test_direct_add<19>(deviceC, deviceB , inlength);

	test_direct_add<20>(deviceC, deviceB , inlength);	
	test_direct_add<21>(deviceC, deviceB , inlength);
	test_direct_add<22>(deviceC, deviceB , inlength);
	test_direct_add<23>(deviceC, deviceB , inlength);
	test_direct_add<24>(deviceC, deviceB , inlength);
	test_direct_add<25>(deviceC, deviceB , inlength);
	test_direct_add<26>(deviceC, deviceB , inlength);
	test_direct_add<27>(deviceC, deviceB , inlength);
	test_direct_add<28>(deviceC, deviceB , inlength);
	test_direct_add<29>(deviceC, deviceB , inlength);
					   	
	test_direct_add<30>(deviceC, deviceB , inlength);
	test_direct_add<31>(deviceC, deviceB , inlength);
	test_direct_add<32>(deviceC, deviceB , inlength);
	test_direct_add<33>(deviceC, deviceB , inlength);
	test_direct_add<34>(deviceC, deviceB , inlength);
	test_direct_add<35>(deviceC, deviceB , inlength);
	test_direct_add<36>(deviceC, deviceB , inlength);
	test_direct_add<37>(deviceC, deviceB , inlength);
	test_direct_add<38>(deviceC, deviceB , inlength);
	test_direct_add<39>(deviceC, deviceB , inlength);
					   	
	test_direct_add<40>(deviceC, deviceB , inlength);
	test_direct_add<41>(deviceC, deviceB , inlength);
	test_direct_add<42>(deviceC, deviceB , inlength);
	test_direct_add<43>(deviceC, deviceB , inlength);
	test_direct_add<44>(deviceC, deviceB , inlength);
	test_direct_add<45>(deviceC, deviceB , inlength);
	test_direct_add<46>(deviceC, deviceB , inlength);
	test_direct_add<47>(deviceC, deviceB , inlength);
	test_direct_add<48>(deviceC, deviceB , inlength);
	test_direct_add<49>(deviceC, deviceB , inlength);
					   	
	test_direct_add<50>(deviceC, deviceB , inlength);
	test_direct_add<51>(deviceC, deviceB , inlength);
	test_direct_add<52>(deviceC, deviceB , inlength);
	test_direct_add<53>(deviceC, deviceB , inlength);
	test_direct_add<54>(deviceC, deviceB , inlength);
	test_direct_add<55>(deviceC, deviceB , inlength);
	test_direct_add<56>(deviceC, deviceB , inlength);
	test_direct_add<57>(deviceC, deviceB , inlength);
	test_direct_add<58>(deviceC, deviceB , inlength);
	test_direct_add<59>(deviceC, deviceB , inlength);
					   	
	test_direct_add<60>(deviceC, deviceB , inlength);
	test_direct_add<61>(deviceC, deviceB , inlength);
	test_direct_add<62>(deviceC, deviceB , inlength);
	test_direct_add<63>(deviceC, deviceB , inlength);


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
		printf(" please input length\n\n");
		return 0;
	}
	unsigned int packedLength = 0;

	packedLength = atoi(argv[1]);

	//packedLength = 256 * 256 * 255;
	test(packedLength);

	return 0;
}




/*

Exmaple result :  MI60-1700Ghz,  800-Mhz
 System minor 0
 System major 3
 agent prop name Vega 20
hip Device prop succeeded 
host allocated 
device allocated 
Host to Device Copied
test_atomic_int, Inlen:[10000000], atomic==2, costs 0.206177 millseconds ,  Bandwidth = 97.003838 Giga test_atomic_int/s
test_atomic_int, Inlen:[10000000], atomic==3, costs 0.283906 millseconds ,  Bandwidth = 105.668800 Giga test_atomic_int/s
test_atomic_int, Inlen:[10000000], atomic==4, costs 0.376051 millseconds ,  Bandwidth = 106.368690 Giga test_atomic_int/s
test_atomic_int, Inlen:[10000000], atomic==5, costs 0.468163 millseconds ,  Bandwidth = 106.800385 Giga test_atomic_int/s
test_atomic_int, Inlen:[10000000], atomic==6, costs 0.560276 millseconds ,  Bandwidth = 107.090134 Giga test_atomic_int/s
test_atomic_int, Inlen:[10000000], atomic==7, costs 0.652692 millseconds ,  Bandwidth = 107.248085 Giga test_atomic_int/s
test_atomic_int, Inlen:[10000000], atomic==8, costs 0.744645 millseconds ,  Bandwidth = 107.433762 Giga test_atomic_int/s
test_atomic_int, Inlen:[10000000], atomic==9, costs 0.836870 millseconds ,  Bandwidth = 107.543648 Giga test_atomic_int/s
test_atomic_int, Inlen:[10000000], atomic==10, costs 0.928982 millseconds ,  Bandwidth = 107.644707 Giga test_atomic_int/s
test_atomic_int, Inlen:[10000000], atomic==11, costs 1.021111 millseconds ,  Bandwidth = 107.725822 Giga test_atomic_int/s
test_atomic_int, Inlen:[10000000], atomic==12, costs 1.114023 millseconds ,  Bandwidth = 107.717659 Giga test_atomic_int/s
test_atomic_int, Inlen:[10000000], atomic==13, costs 1.205592 millseconds ,  Bandwidth = 107.830849 Giga test_atomic_int/s
test_atomic_int, Inlen:[10000000], atomic==14, costs 1.297673 millseconds ,  Bandwidth = 107.885468 Giga test_atomic_int/s
test_atomic_int, Inlen:[10000000], atomic==15, costs 1.390089 millseconds ,  Bandwidth = 107.906746 Giga test_atomic_int/s
test_atomic_int, Inlen:[10000000], atomic==16, costs 1.482218 millseconds ,  Bandwidth = 107.946358 Giga test_atomic_int/s
test_atomic_int, Inlen:[10000000], atomic==17, costs 1.574331 millseconds ,  Bandwidth = 107.982407 Giga test_atomic_int/s
test_atomic_int, Inlen:[10000000], atomic==18, costs 1.666507 millseconds ,  Bandwidth = 108.010353 Giga test_atomic_int/s
test_atomic_int, Inlen:[10000000], atomic==19, costs 1.758764 millseconds ,  Bandwidth = 108.030434 Giga test_atomic_int/s
test_atomic_int, Inlen:[10000000], atomic==20, costs 1.850876 millseconds ,  Bandwidth = 108.056915 Giga test_atomic_int/s
test_atomic_int, Inlen:[10000000], atomic==21, costs 1.943037 millseconds ,  Bandwidth = 108.078239 Giga test_atomic_int/s
test_atomic_int, Inlen:[10000000], atomic==22, costs 2.035501 millseconds ,  Bandwidth = 108.081474 Giga test_atomic_int/s
test_atomic_int, Inlen:[10000000], atomic==23, costs 2.129198 millseconds ,  Bandwidth = 108.021889 Giga test_atomic_int/s
test_atomic_int, Inlen:[10000000], atomic==24, costs 2.221599 millseconds ,  Bandwidth = 108.030312 Giga test_atomic_int/s
test_atomic_int, Inlen:[10000000], atomic==25, costs 2.312543 millseconds ,  Bandwidth = 108.106094 Giga test_atomic_int/s
test_atomic_int, Inlen:[10000000], atomic==26, costs 2.404560 millseconds ,  Bandwidth = 108.127892 Giga test_atomic_int/s
test_atomic_int, Inlen:[10000000], atomic==27, costs 2.496785 millseconds ,  Bandwidth = 108.139084 Giga test_atomic_int/s
test_atomic_int, Inlen:[10000000], atomic==28, costs 2.588993 millseconds ,  Bandwidth = 108.150146 Giga test_atomic_int/s
test_atomic_int, Inlen:[10000000], atomic==29, costs 2.681138 millseconds ,  Bandwidth = 108.163033 Giga test_atomic_int/s
test_atomic_int, Inlen:[10000000], atomic==30, costs 3.372311 millseconds ,  Bandwidth = 88.959778 Giga test_atomic_int/s
test_atomic_int, Inlen:[10000000], atomic==31, costs 2.865875 millseconds ,  Bandwidth = 108.169411 Giga test_atomic_int/s
test_atomic_int, Inlen:[10000000], atomic==32, costs 2.958051 millseconds ,  Bandwidth = 108.179321 Giga test_atomic_int/s
test_atomic_int, Inlen:[10000000], atomic==33, costs 3.050164 millseconds ,  Bandwidth = 108.190903 Giga test_atomic_int/s
test_atomic_int, Inlen:[10000000], atomic==34, costs 3.142325 millseconds ,  Bandwidth = 108.200142 Giga test_atomic_int/s
test_atomic_int, Inlen:[10000000], atomic==35, costs 3.234549 millseconds ,  Bandwidth = 108.206726 Giga test_atomic_int/s
test_atomic_int, Inlen:[10000000], atomic==36, costs 3.326918 millseconds ,  Bandwidth = 108.208260 Giga test_atomic_int/s
test_atomic_int, Inlen:[10000000], atomic==37, costs 3.418999 millseconds ,  Bandwidth = 108.218819 Giga test_atomic_int/s
test_atomic_int, Inlen:[10000000], atomic==38, costs 3.511111 millseconds ,  Bandwidth = 108.227837 Giga test_atomic_int/s
test_atomic_int, Inlen:[10000000], atomic==39, costs 3.603272 millseconds ,  Bandwidth = 108.234955 Giga test_atomic_int/s
test_atomic_int, Inlen:[10000000], atomic==40, costs 3.695513 millseconds ,  Bandwidth = 108.239380 Giga test_atomic_int/s
test_atomic_int, Inlen:[10000000], atomic==41, costs 3.787641 millseconds ,  Bandwidth = 108.246796 Giga test_atomic_int/s
test_atomic_int, Inlen:[10000000], atomic==42, costs 3.879994 millseconds ,  Bandwidth = 108.247589 Giga test_atomic_int/s
test_atomic_int, Inlen:[10000000], atomic==43, costs 3.972139 millseconds ,  Bandwidth = 108.254028 Giga test_atomic_int/s
test_atomic_int, Inlen:[10000000], atomic==44, costs 4.064411 millseconds ,  Bandwidth = 108.256760 Giga test_atomic_int/s
test_atomic_int, Inlen:[10000000], atomic==45, costs 4.156476 millseconds ,  Bandwidth = 108.264801 Giga test_atomic_int/s
test_atomic_int, Inlen:[10000000], atomic==46, costs 4.248733 millseconds ,  Bandwidth = 108.267586 Giga test_atomic_int/s
test_atomic_int, Inlen:[10000000], atomic==47, costs 4.340909 millseconds ,  Bandwidth = 108.272255 Giga test_atomic_int/s
test_atomic_int, Inlen:[10000000], atomic==48, costs 4.433261 millseconds ,  Bandwidth = 108.272430 Giga test_atomic_int/s
test_atomic_int, Inlen:[10000000], atomic==49, costs 4.525358 millseconds ,  Bandwidth = 108.278732 Giga test_atomic_int/s
test_atomic_int, Inlen:[10000000], atomic==50, costs 4.617423 millseconds ,  Bandwidth = 108.285507 Giga test_atomic_int/s
test_atomic_int, Inlen:[10000000], atomic==51, costs 4.709615 millseconds ,  Bandwidth = 108.289101 Giga test_atomic_int/s
test_atomic_int, Inlen:[10000000], atomic==52, costs 4.801872 millseconds ,  Bandwidth = 108.291100 Giga test_atomic_int/s
test_atomic_int, Inlen:[10000000], atomic==53, costs 4.894145 millseconds ,  Bandwidth = 108.292671 Giga test_atomic_int/s
test_atomic_int, Inlen:[10000000], atomic==54, costs 4.986337 millseconds ,  Bandwidth = 108.295929 Giga test_atomic_int/s
test_atomic_int, Inlen:[10000000], atomic==55, costs 5.078418 millseconds ,  Bandwidth = 108.301453 Giga test_atomic_int/s
test_atomic_int, Inlen:[10000000], atomic==56, costs 5.170562 millseconds ,  Bandwidth = 108.305435 Giga test_atomic_int/s
test_atomic_int, Inlen:[10000000], atomic==57, costs 5.262771 millseconds ,  Bandwidth = 108.307968 Giga test_atomic_int/s
test_atomic_int, Inlen:[10000000], atomic==58, costs 5.354948 millseconds ,  Bandwidth = 108.311050 Giga test_atomic_int/s
test_atomic_int, Inlen:[10000000], atomic==59, costs 5.447220 millseconds ,  Bandwidth = 108.312119 Giga test_atomic_int/s
test_atomic_int, Inlen:[10000000], atomic==60, costs 5.539509 millseconds ,  Bandwidth = 108.312859 Giga test_atomic_int/s
test_atomic_int, Inlen:[10000000], atomic==61, costs 5.631989 millseconds ,  Bandwidth = 108.309860 Giga test_atomic_int/s
test_atomic_int, Inlen:[10000000], atomic==62, costs 5.723814 millseconds ,  Bandwidth = 108.319382 Giga test_atomic_int/s
test_atomic_int, Inlen:[10000000], atomic==63, costs 5.816054 millseconds ,  Bandwidth = 108.320854 Giga test_atomic_int/s
test_atomic_float, Inlen:[10000000], atomic==2, costs 0.438419 millseconds ,  Bandwidth = 45.618458 Giga atomic_add_float/s
test_atomic_float, Inlen:[10000000], atomic==3, costs 0.598244 millseconds ,  Bandwidth = 50.146770 Giga atomic_add_float/s
test_atomic_float, Inlen:[10000000], atomic==4, costs 1.447514 millseconds ,  Bandwidth = 27.633591 Giga atomic_add_float/s
test_atomic_float, Inlen:[10000000], atomic==5, costs 1.426458 millseconds ,  Bandwidth = 35.051868 Giga atomic_add_float/s
test_atomic_float, Inlen:[10000000], atomic==6, costs 2.002605 millseconds ,  Bandwidth = 29.960972 Giga atomic_add_float/s
test_atomic_float, Inlen:[10000000], atomic==7, costs 2.648946 millseconds ,  Bandwidth = 26.425610 Giga atomic_add_float/s
test_atomic_float, Inlen:[10000000], atomic==8, costs 3.354839 millseconds ,  Bandwidth = 23.846153 Giga atomic_add_float/s
test_atomic_float, Inlen:[10000000], atomic==9, costs 4.067707 millseconds ,  Bandwidth = 22.125488 Giga atomic_add_float/s
test_atomic_float, Inlen:[10000000], atomic==10, costs 4.949585 millseconds ,  Bandwidth = 20.203714 Giga atomic_add_float/s
test_atomic_float, Inlen:[10000000], atomic==11, costs 5.872215 millseconds ,  Bandwidth = 18.732283 Giga atomic_add_float/s
test_atomic_float, Inlen:[10000000], atomic==12, costs 6.910238 millseconds ,  Bandwidth = 17.365538 Giga atomic_add_float/s
test_atomic_float, Inlen:[10000000], atomic==13, costs 7.938901 millseconds ,  Bandwidth = 16.375063 Giga atomic_add_float/s
test_atomic_float, Inlen:[10000000], atomic==14, costs 9.063004 millseconds ,  Bandwidth = 15.447416 Giga atomic_add_float/s
test_atomic_float, Inlen:[10000000], atomic==15, costs 10.186228 millseconds ,  Bandwidth = 14.725766 Giga atomic_add_float/s
test_atomic_float, Inlen:[10000000], atomic==16, costs 11.427197 millseconds ,  Bandwidth = 14.001684 Giga atomic_add_float/s
test_atomic_float, Inlen:[10000000], atomic==17, costs 12.773990 millseconds ,  Bandwidth = 13.308293 Giga atomic_add_float/s
test_atomic_float, Inlen:[10000000], atomic==18, costs 14.198318 millseconds ,  Bandwidth = 12.677558 Giga atomic_add_float/s
test_atomic_float, Inlen:[10000000], atomic==19, costs 15.685353 millseconds ,  Bandwidth = 12.113212 Giga atomic_add_float/s
test_atomic_float, Inlen:[10000000], atomic==20, costs 17.261829 millseconds ,  Bandwidth = 11.586258 Giga atomic_add_float/s
test_atomic_float, Inlen:[10000000], atomic==21, costs 18.841263 millseconds ,  Bandwidth = 11.145749 Giga atomic_add_float/s
test_atomic_float, Inlen:[10000000], atomic==22, costs 20.512842 millseconds ,  Bandwidth = 10.724989 Giga atomic_add_float/s
test_atomic_float, Inlen:[10000000], atomic==23, costs 22.281702 millseconds ,  Bandwidth = 10.322371 Giga atomic_add_float/s
test_atomic_float, Inlen:[10000000], atomic==24, costs 24.155043 millseconds ,  Bandwidth = 9.935813 Giga atomic_add_float/s
test_atomic_float, Inlen:[10000000], atomic==25, costs 26.004400 millseconds ,  Bandwidth = 9.613757 Giga atomic_add_float/s
test_atomic_float, Inlen:[10000000], atomic==26, costs 28.601282 millseconds ,  Bandwidth = 9.090501 Giga atomic_add_float/s
test_atomic_float, Inlen:[10000000], atomic==27, costs 29.888523 millseconds ,  Bandwidth = 9.033568 Giga atomic_add_float/s
test_atomic_float, Inlen:[10000000], atomic==28, costs 31.840042 millseconds ,  Bandwidth = 8.793959 Giga atomic_add_float/s
test_atomic_float, Inlen:[10000000], atomic==29, costs 33.911671 millseconds ,  Bandwidth = 8.551628 Giga atomic_add_float/s
test_atomic_float, Inlen:[10000000], atomic==30, costs 35.952118 millseconds ,  Bandwidth = 8.344432 Giga atomic_add_float/s
test_atomic_float, Inlen:[10000000], atomic==31, costs 38.829529 millseconds ,  Bandwidth = 7.983615 Giga atomic_add_float/s
test_atomic_float, Inlen:[10000000], atomic==32, costs 40.347862 millseconds ,  Bandwidth = 7.931027 Giga atomic_add_float/s
test_atomic_float, Inlen:[10000000], atomic==33, costs 42.721794 millseconds ,  Bandwidth = 7.724395 Giga atomic_add_float/s
test_atomic_float, Inlen:[10000000], atomic==34, costs 45.124260 millseconds ,  Bandwidth = 7.534750 Giga atomic_add_float/s
test_atomic_float, Inlen:[10000000], atomic==35, costs 48.241116 millseconds ,  Bandwidth = 7.255222 Giga atomic_add_float/s
test_atomic_float, Inlen:[10000000], atomic==36, costs 50.089241 millseconds ,  Bandwidth = 7.187172 Giga atomic_add_float/s
test_atomic_float, Inlen:[10000000], atomic==37, costs 52.787518 millseconds ,  Bandwidth = 7.009233 Giga atomic_add_float/s
test_atomic_float, Inlen:[10000000], atomic==38, costs 55.493427 millseconds ,  Bandwidth = 6.847658 Giga atomic_add_float/s
test_atomic_float, Inlen:[10000000], atomic==39, costs 58.843452 millseconds ,  Bandwidth = 6.627755 Giga atomic_add_float/s
test_atomic_float, Inlen:[10000000], atomic==40, costs 60.979420 millseconds ,  Bandwidth = 6.559590 Giga atomic_add_float/s
test_atomic_float, Inlen:[10000000], atomic==41, costs 63.931324 millseconds ,  Bandwidth = 6.413132 Giga atomic_add_float/s
test_atomic_float, Inlen:[10000000], atomic==42, costs 66.921188 millseconds ,  Bandwidth = 6.276039 Giga atomic_add_float/s
test_atomic_float, Inlen:[10000000], atomic==43, costs 70.579483 millseconds ,  Bandwidth = 6.092422 Giga atomic_add_float/s
test_atomic_float, Inlen:[10000000], atomic==44, costs 73.023651 millseconds ,  Bandwidth = 6.025445 Giga atomic_add_float/s
test_atomic_float, Inlen:[10000000], atomic==45, costs 76.271141 millseconds ,  Bandwidth = 5.900003 Giga atomic_add_float/s
test_atomic_float, Inlen:[10000000], atomic==46, costs 79.573532 millseconds ,  Bandwidth = 5.780817 Giga atomic_add_float/s
test_atomic_float, Inlen:[10000000], atomic==47, costs 83.572609 millseconds ,  Bandwidth = 5.623852 Giga atomic_add_float/s
test_atomic_float, Inlen:[10000000], atomic==48, costs 86.317413 millseconds ,  Bandwidth = 5.560871 Giga atomic_add_float/s
test_atomic_float, Inlen:[10000000], atomic==49, costs 89.913918 millseconds ,  Bandwidth = 5.449657 Giga atomic_add_float/s
test_atomic_float, Inlen:[10000000], atomic==50, costs 93.491562 millseconds ,  Bandwidth = 5.348076 Giga atomic_add_float/s
test_atomic_float, Inlen:[10000000], atomic==51, costs 97.141678 millseconds ,  Bandwidth = 5.250063 Giga atomic_add_float/s
test_atomic_float, Inlen:[10000000], atomic==52, costs 101.480469 millseconds ,  Bandwidth = 5.124138 Giga atomic_add_float/s
test_atomic_float, Inlen:[10000000], atomic==53, costs 104.674301 millseconds ,  Bandwidth = 5.063324 Giga atomic_add_float/s
test_atomic_float, Inlen:[10000000], atomic==54, costs 108.542099 millseconds ,  Bandwidth = 4.975029 Giga atomic_add_float/s
test_atomic_float, Inlen:[10000000], atomic==55, costs 112.438614 millseconds ,  Bandwidth = 4.891558 Giga atomic_add_float/s
test_atomic_float, Inlen:[10000000], atomic==56, costs 116.431618 millseconds ,  Bandwidth = 4.809690 Giga atomic_add_float/s
test_atomic_float, Inlen:[10000000], atomic==57, costs 121.225609 millseconds ,  Bandwidth = 4.701977 Giga atomic_add_float/s
test_atomic_float, Inlen:[10000000], atomic==58, costs 124.773087 millseconds ,  Bandwidth = 4.648438 Giga atomic_add_float/s
test_atomic_float, Inlen:[10000000], atomic==59, costs 128.993332 millseconds ,  Bandwidth = 4.573880 Giga atomic_add_float/s
test_atomic_float, Inlen:[10000000], atomic==60, costs 133.265259 millseconds ,  Bandwidth = 4.502299 Giga atomic_add_float/s
test_atomic_float, Inlen:[10000000], atomic==61, costs 137.754608 millseconds ,  Bandwidth = 4.428164 Giga atomic_add_float/s
test_atomic_float, Inlen:[10000000], atomic==62, costs 142.920654 millseconds ,  Bandwidth = 4.338071 Giga atomic_add_float/s
test_atomic_float, Inlen:[10000000], atomic==63, costs 146.828690 millseconds ,  Bandwidth = 4.290715 Giga atomic_add_float/s
Float_direct_reduction, Inlen:[10000000], atomic==2, costs 0.202449 millseconds ,  Bandwidth = 98.790115 Giga Float_direct_reduction/s
Float_direct_reduction, Inlen:[10000000], atomic==3, costs 0.286274 millseconds ,  Bandwidth = 104.794701 Giga Float_direct_reduction/s
Float_direct_reduction, Inlen:[10000000], atomic==4, costs 0.376691 millseconds ,  Bandwidth = 106.187920 Giga Float_direct_reduction/s
Float_direct_reduction, Inlen:[10000000], atomic==5, costs 0.468803 millseconds ,  Bandwidth = 106.654518 Giga Float_direct_reduction/s
Float_direct_reduction, Inlen:[10000000], atomic==6, costs 0.560980 millseconds ,  Bandwidth = 106.955666 Giga Float_direct_reduction/s
Float_direct_reduction, Inlen:[10000000], atomic==7, costs 0.653381 millseconds ,  Bandwidth = 107.135094 Giga Float_direct_reduction/s
Float_direct_reduction, Inlen:[10000000], atomic==8, costs 0.745797 millseconds ,  Bandwidth = 107.267738 Giga Float_direct_reduction/s
Float_direct_reduction, Inlen:[10000000], atomic==9, costs 0.838134 millseconds ,  Bandwidth = 107.381386 Giga Float_direct_reduction/s
Float_direct_reduction, Inlen:[10000000], atomic==10, costs 0.930231 millseconds ,  Bandwidth = 107.500206 Giga Float_direct_reduction/s
Float_direct_reduction, Inlen:[10000000], atomic==11, costs 1.022615 millseconds ,  Bandwidth = 107.567314 Giga Float_direct_reduction/s
Float_direct_reduction, Inlen:[10000000], atomic==12, costs 1.114952 millseconds ,  Bandwidth = 107.627937 Giga Float_direct_reduction/s
Float_direct_reduction, Inlen:[10000000], atomic==13, costs 1.206921 millseconds ,  Bandwidth = 107.712112 Giga Float_direct_reduction/s
Float_direct_reduction, Inlen:[10000000], atomic==14, costs 1.299193 millseconds ,  Bandwidth = 107.759171 Giga Float_direct_reduction/s
Float_direct_reduction, Inlen:[10000000], atomic==15, costs 1.391610 millseconds ,  Bandwidth = 107.788803 Giga Float_direct_reduction/s
Float_direct_reduction, Inlen:[10000000], atomic==16, costs 1.484907 millseconds ,  Bandwidth = 107.750870 Giga Float_direct_reduction/s
Float_direct_reduction, Inlen:[10000000], atomic==17, costs 1.577244 millseconds ,  Bandwidth = 107.782974 Giga Float_direct_reduction/s
Float_direct_reduction, Inlen:[10000000], atomic==18, costs 1.669996 millseconds ,  Bandwidth = 107.784676 Giga Float_direct_reduction/s
Float_direct_reduction, Inlen:[10000000], atomic==19, costs 1.762349 millseconds ,  Bandwidth = 107.810669 Giga Float_direct_reduction/s
Float_direct_reduction, Inlen:[10000000], atomic==20, costs 1.854430 millseconds ,  Bandwidth = 107.849876 Giga Float_direct_reduction/s
Float_direct_reduction, Inlen:[10000000], atomic==21, costs 1.946798 millseconds ,  Bandwidth = 107.869423 Giga Float_direct_reduction/s
Float_direct_reduction, Inlen:[10000000], atomic==22, costs 2.039535 millseconds ,  Bandwidth = 107.867722 Giga Float_direct_reduction/s
Float_direct_reduction, Inlen:[10000000], atomic==23, costs 2.133647 millseconds ,  Bandwidth = 107.796631 Giga Float_direct_reduction/s
Float_direct_reduction, Inlen:[10000000], atomic==24, costs 2.226880 millseconds ,  Bandwidth = 107.774086 Giga Float_direct_reduction/s
Float_direct_reduction, Inlen:[10000000], atomic==25, costs 2.318049 millseconds ,  Bandwidth = 107.849319 Giga Float_direct_reduction/s
Float_direct_reduction, Inlen:[10000000], atomic==26, costs 2.410930 millseconds ,  Bandwidth = 107.842209 Giga Float_direct_reduction/s
Float_direct_reduction, Inlen:[10000000], atomic==27, costs 2.504194 millseconds ,  Bandwidth = 107.819115 Giga Float_direct_reduction/s
Float_direct_reduction, Inlen:[10000000], atomic==28, costs 2.597267 millseconds ,  Bandwidth = 107.805634 Giga Float_direct_reduction/s
Float_direct_reduction, Inlen:[10000000], atomic==29, costs 2.689956 millseconds ,  Bandwidth = 107.808464 Giga Float_direct_reduction/s
Float_direct_reduction, Inlen:[10000000], atomic==30, costs 2.782692 millseconds ,  Bandwidth = 107.809258 Giga Float_direct_reduction/s
Float_direct_reduction, Inlen:[10000000], atomic==31, costs 2.874789 millseconds ,  Bandwidth = 107.834000 Giga Float_direct_reduction/s
Float_direct_reduction, Inlen:[10000000], atomic==32, costs 2.968518 millseconds ,  Bandwidth = 107.797905 Giga Float_direct_reduction/s
Float_direct_reduction, Inlen:[10000000], atomic==33, costs 3.058454 millseconds ,  Bandwidth = 107.897644 Giga Float_direct_reduction/s
Float_direct_reduction, Inlen:[10000000], atomic==34, costs 3.148199 millseconds ,  Bandwidth = 107.998253 Giga Float_direct_reduction/s
Float_direct_reduction, Inlen:[10000000], atomic==35, costs 3.240360 millseconds ,  Bandwidth = 108.012695 Giga Float_direct_reduction/s
Float_direct_reduction, Inlen:[10000000], atomic==36, costs 3.333096 millseconds ,  Bandwidth = 108.007683 Giga Float_direct_reduction/s
Float_direct_reduction, Inlen:[10000000], atomic==37, costs 3.425833 millseconds ,  Bandwidth = 108.002922 Giga Float_direct_reduction/s
Float_direct_reduction, Inlen:[10000000], atomic==38, costs 3.518906 millseconds ,  Bandwidth = 107.988121 Giga Float_direct_reduction/s
Float_direct_reduction, Inlen:[10000000], atomic==39, costs 3.613387 millseconds ,  Bandwidth = 107.931984 Giga Float_direct_reduction/s
Float_direct_reduction, Inlen:[10000000], atomic==40, costs 3.711083 millseconds ,  Bandwidth = 107.785248 Giga Float_direct_reduction/s
Float_direct_reduction, Inlen:[10000000], atomic==41, costs 3.829964 millseconds ,  Bandwidth = 107.050606 Giga Float_direct_reduction/s
Float_direct_reduction, Inlen:[10000000], atomic==42, costs 3.958909 millseconds ,  Bandwidth = 106.089836 Giga Float_direct_reduction/s
Float_direct_reduction, Inlen:[10000000], atomic==43, costs 4.094526 millseconds ,  Bandwidth = 105.018265 Giga Float_direct_reduction/s
Float_direct_reduction, Inlen:[10000000], atomic==44, costs 4.231663 millseconds ,  Bandwidth = 103.978043 Giga Float_direct_reduction/s
Float_direct_reduction, Inlen:[10000000], atomic==45, costs 4.377632 millseconds ,  Bandwidth = 102.795296 Giga Float_direct_reduction/s
Float_direct_reduction, Inlen:[10000000], atomic==46, costs 4.519249 millseconds ,  Bandwidth = 101.786827 Giga Float_direct_reduction/s
Float_direct_reduction, Inlen:[10000000], atomic==47, costs 4.668850 millseconds ,  Bandwidth = 100.667191 Giga Float_direct_reduction/s
Float_direct_reduction, Inlen:[10000000], atomic==48, costs 4.823187 millseconds ,  Bandwidth = 99.519249 Giga Float_direct_reduction/s
Float_direct_reduction, Inlen:[10000000], atomic==49, costs 4.976980 millseconds ,  Bandwidth = 98.453278 Giga Float_direct_reduction/s
Float_direct_reduction, Inlen:[10000000], atomic==50, costs 5.133958 millseconds ,  Bandwidth = 97.390747 Giga Float_direct_reduction/s
Float_direct_reduction, Inlen:[10000000], atomic==51, costs 5.294247 millseconds ,  Bandwidth = 96.330986 Giga Float_direct_reduction/s
Float_direct_reduction, Inlen:[10000000], atomic==52, costs 5.452888 millseconds ,  Bandwidth = 95.362312 Giga Float_direct_reduction/s
Float_direct_reduction, Inlen:[10000000], atomic==53, costs 5.604169 millseconds ,  Bandwidth = 94.572449 Giga Float_direct_reduction/s
Float_direct_reduction, Inlen:[10000000], atomic==54, costs 5.736986 millseconds ,  Bandwidth = 94.126083 Giga Float_direct_reduction/s
Float_direct_reduction, Inlen:[10000000], atomic==55, costs 5.946876 millseconds ,  Bandwidth = 92.485542 Giga Float_direct_reduction/s
Float_direct_reduction, Inlen:[10000000], atomic==56, costs 6.135725 millseconds ,  Bandwidth = 91.268761 Giga Float_direct_reduction/s
Float_direct_reduction, Inlen:[10000000], atomic==57, costs 6.295006 millseconds ,  Bandwidth = 90.547966 Giga Float_direct_reduction/s
Float_direct_reduction, Inlen:[10000000], atomic==58, costs 6.462831 millseconds ,  Bandwidth = 89.743950 Giga Float_direct_reduction/s
Float_direct_reduction, Inlen:[10000000], atomic==59, costs 6.631664 millseconds ,  Bandwidth = 88.967102 Giga Float_direct_reduction/s
Float_direct_reduction, Inlen:[10000000], atomic==60, costs 6.816530 millseconds ,  Bandwidth = 88.021332 Giga Float_direct_reduction/s
Float_direct_reduction, Inlen:[10000000], atomic==61, costs 7.002419 millseconds ,  Bandwidth = 87.112747 Giga Float_direct_reduction/s
Float_direct_reduction, Inlen:[10000000], atomic==62, costs 7.190980 millseconds ,  Bandwidth = 86.219116 Giga Float_direct_reduction/s
Float_direct_reduction, Inlen:[10000000], atomic==63, costs 7.382278 millseconds ,  Bandwidth = 85.339508 Giga Float_direct_reduction/s

*/