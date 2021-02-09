#include <assert.h>
#include <stdio.h>
#include <algorithm>
#include <stdlib.h>
#include<iostream>
#include "hip/hip_runtime.h"


#define HIP_ASSERT(x) (assert((x)==hipSuccess))


#define WIDTH     4096
#define HEIGHT    1024

#define NUM       (WIDTH*HEIGHT+2048)

#define THREADS_PER_BLOCK_X  256
#define THREADS_PER_BLOCK_Y  1
#define THREADS_PER_BLOCK_Z  1

__global__ void conv1d_2048(const float* a, const float* w, float* __restrict__ r ){
  int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
  int gid = x;
  float sum =0;
  for(int i=0; i < 2048; i++)     sum += a[gid + i] * w [i];  
  r[gid] =sum;
}


__global__ void conv1d_2048_opt1(const float* a, const float* w, float* __restrict__ r){
  int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
  int gid = x * 4;
  if( gid >= (WIDTH*HEIGHT)){
     return ;
  }
  float sum[4];
  float ww[8];
  float aa[4];
  for(int i=0; i <4; i++)
  {
      sum[i] = 0;
      ww[i] = w[i];
      aa[i] = a[i+gid];

  }

  for(int i=0; i < 4; i++)
  {
      for(int j=0; j < (4-i);j++)
      {
         sum[i] += aa[j+i] * ww[j];
      }
       
  }

  for(int i=4; i < 8; i+=4)
  {
     for(int s=0; s<4; s++)
     {
       aa[s] = a[gid+i+s];
       ww[s+4] = w[i+s];
     }
     for(int s =0; s < 4; s++)
     {
        int offset = 4-s;
        for(int t=0; t < 4; t++)
        {
          sum[s] += aa[t] * ww[offset + t];
        }
     }

     //shift left
     for(int s=0; s < 4; s++){
       ww[s] = ww[s+4];
     }
     
  }


  for(int i=8; i < 2048; i+=4)
  {
     for(int s=0; s<4; s++)
     {
       aa[s] = a[gid+i+s];
       ww[s+4] = w[i+s];
     }
     for(int s =0; s < 4; s++)
     {
        int offset = 4-s;
        for(int t=0; t < 4; t++)
        {
          sum[s] += aa[t] * ww[offset + t];
        }
     }

     //shift left

     for(int s=0; s < 4; s++){
       ww[s] = ww[s+4];
     }


     i+=4;
     for(int s=0; s<4; s++)
     {
       aa[s] = a[gid+i+s];
       ww[s+4] = w[i+s];
     }
     for(int s =0; s < 4; s++)
     {
        int offset = 4-s;
        for(int t=0; t < 4; t++)
        {
          sum[s] += aa[t] * ww[offset + t];
        }
     }

     //shift left
     for(int s=0; s < 4; s++){
       ww[s] = ww[s+4];
     }
     
  }

  //last 3
  for(int i=0; i <3; i++){
    aa[i] = a[gid+2048+i];   
  }
#if 0  
  sum[1] += aa[0] * ww[3];
  sum[2] += aa[0] * ww[2];
  sum[2] += aa[1] * ww[3];
  sum[3] += aa[0] * ww[1];
  sum[3] += aa[1] * ww[2];
  sum[3] += aa[2] * ww[3];
#else 
  for(int s=0; s < 3; s++)
  {
      for(int t=0; t <(s+1); t++)
      {
        sum[s+1] += aa[t] * ww[3-s+t];
      }
  }
#endif 

   for(int i=0; i <4; i++){
       r[gid+i] = sum[i];
   } 
}


__global__ void conv1d_2048_opt2(const float* a, const float* w, float* __restrict__ r){
  int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
  int gid = x * 4;
  if( gid >= (WIDTH*HEIGHT)){
     return ;
  }


  float sum[4];
  float ww[8];
  float aa[4];
  for(int i=0; i <4; i++)
  {
      sum[i] = 0;
      ww[i] = w[i];
      aa[i] = a[i+gid];

  }

  for(int i=0; i < 4; i++)
  {
      for(int j=0; j < (4-i);j++)
      {
         sum[i] += aa[j+i] * ww[j];
      }
       
  }

  for(int i=4; i < 8; i+=4)
  {
     for(int s=0; s<4; s++)
     {
       aa[s] = a[gid+i+s];
       ww[s+4] = w[i+s];
     }
     for(int s =0; s < 4; s++)
     {
        int offset = 4-s;
        for(int t=0; t < 4; t++)
        {
          sum[s] += aa[t] * ww[offset + t];
        }
     }

     //shift left
     for(int s=0; s < 4; s++){
       ww[s] = ww[s+4];
     }
     
  }

#pragma unroll
  for(int i=8; i < 2048; i+=4)
  {
     for(int s=0; s<4; s++)
     {
       aa[s] = a[gid+i+s];
       ww[s+4] = w[i+s];
     }
     for(int s =0; s < 4; s++)
     {
        int offset = 4-s;
        for(int t=0; t < 4; t++)
        {
          sum[s] += aa[t] * ww[offset + t];
        }
     }

     //shift left

     for(int s=0; s < 4; s++){
       ww[s] = ww[s+4];
     }


     i+=4;
     for(int s=0; s<4; s++)
     {
       aa[s] = a[gid+i+s];
       ww[s+4] = w[i+s];
     }
     for(int s =0; s < 4; s++)
     {
        int offset = 4-s;
        for(int t=0; t < 4; t++)
        {
          sum[s] += aa[t] * ww[offset + t];
        }
     }

     //shift left
     for(int s=0; s < 4; s++){
       ww[s] = ww[s+4];
     }
     
  }

  //last 3
  for(int i=0; i <3; i++){
    aa[i] = a[gid+2048+i];   
  }
#if 0  
  sum[1] += aa[0] * ww[3];
  sum[2] += aa[0] * ww[2];
  sum[2] += aa[1] * ww[3];
  sum[3] += aa[0] * ww[1];
  sum[3] += aa[1] * ww[2];
  sum[3] += aa[2] * ww[3];
#else 
  for(int s=0; s < 3; s++)
  {
      for(int t=0; t <(s+1); t++)
      {
        sum[s+1] += aa[t] * ww[3-s+t];
      }
  }
#endif 

   for(int i=0; i <4; i++){
       r[gid+i] = sum[i];
   } 
}

//workgroup size = 256
//LOAD all data into LDS 
//LDS provides 2X bandwidth vs L1 Cache

__global__ void conv1d_2048_opt3(const float* a, const float* w, float* __restrict__ r){
  int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
  int gid = x * 4;
  if( gid >= (WIDTH*HEIGHT)){
     return ;
  }
  float sum[4];
  float ww[8];
  float aa[4];
  const int F_SIZE = 2048;
  const int WK_SIZE = 256;
  const int OUT_PIX_SIZE = 4;
  __shared__ float aData[F_SIZE+WK_SIZE+OUT_PIX_SIZE]; 

  //use global load dword x4
  for(int i=0; i < (F_SIZE); i+=256){
#if 0           
      aData[i + hipThreadIdx_x * 4 + 0 ] = a[gid + i + hipThreadIdx_x * 4 + 0];
      aData[i + hipThreadIdx_x * 4 + 1 ] = a[gid + i + hipThreadIdx_x * 4 + 1];
      aData[i + hipThreadIdx_x * 4 + 2 ] = a[gid + i + hipThreadIdx_x * 4 + 2];
      aData[i + hipThreadIdx_x * 4 + 3 ] = a[gid + i + hipThreadIdx_x * 4 + 3];
#else
    aData[i + hipThreadIdx_x]  = a[gid + i + hipThreadIdx_x];
#endif 
  }

  // aData[F_SIZE + hipThreadIdx_x] = a [F_SIZE + hipThreadIdx_x];
  //last 4 data 
  if(hipThreadIdx_x < 4){
    aData[hipThreadIdx_x + F_SIZE+WK_SIZE] =  a[gid + hipThreadIdx_x + F_SIZE+WK_SIZE];
  }

  for(int i=0; i <4; i++)
  {
      sum[i] = 0;
      ww[i] = w[i];
      //aa[i] = a[i+gid];
      aa[i] = aData[i+hipThreadIdx_x];

  }

  //first 8x weight for 1st output pixel
  //to make sure filter ready
  for(int i=0; i < 4; i++)
  {
      for(int j=0; j < (4-i);j++)
      {
         sum[i] += aa[j+i] * ww[j];
      }
       
  }

  for(int i=4; i < 8; i+=4)
  {
     for(int s=0; s<4; s++)
     {
       //aa[s] = a[gid+i+s];
       aa[s] = aData[hipThreadIdx_x+i+s];
       ww[s+4] = w[i+s];
     }
     for(int s =0; s < 4; s++)
     {
        int offset = 4-s;
        for(int t=0; t < 4; t++)
        {
          sum[s] += aa[t] * ww[offset + t];
        }
     }

     //shift left
     for(int s=0; s < 4; s++){
       ww[s] = ww[s+4];
     }
     
  }

  //first 2040x weight for all output pixel
#pragma unroll
  for(int i=8; i < 2048; i+=4)
  {
     for(int s=0; s<4; s++)
     {
       //aa[s] = a[gid+i+s];
       aa[s] = aData[hipThreadIdx_x+i+s];
       ww[s+4] = w[i+s];
     }
     for(int s =0; s < 4; s++)
     {
        int offset = 4-s;
        for(int t=0; t < 4; t++)
        {
          sum[s] += aa[t] * ww[offset + t];
        }
     }

     //shift left

     for(int s=0; s < 4; s++){
       ww[s] = ww[s+4];
     }


     i+=4;
     for(int s=0; s<4; s++)
     {
       //aa[s] = a[gid+i+s];
       aa[s] = aData[hipThreadIdx_x + i + s];
       ww[s+4] = w[i+s];
     }
     for(int s =0; s < 4; s++)
     {
        int offset = 4-s;
        for(int t=0; t < 4; t++)
        {
          sum[s] += aa[t] * ww[offset + t];
        }
     }

     //shift left
     for(int s=0; s < 4; s++){
       ww[s] = ww[s+4];
     }     
  }

  //rest 3 pixels's rest weight, 
  for(int i=0; i <3; i++){
    aa[i] = aData[hipThreadIdx_x + F_SIZE + i];   
  }
#if 0  
  sum[1] += aa[0] * ww[3];
  sum[2] += aa[0] * ww[2];
  sum[2] += aa[1] * ww[3];
  sum[3] += aa[0] * ww[1];
  sum[3] += aa[1] * ww[2];
  sum[3] += aa[2] * ww[3];
#else 
  for(int s=0; s < 3; s++)
  {
      for(int t=0; t <(s+1); t++)
      {
        sum[s+1] += aa[t] * ww[3-s+t];
      }
  }
#endif 

   for(int i=0; i <4; i++){
       r[gid+i] = sum[i];
   } 
}


//BLOCK : Result  tile: 8x64
//Total waves can be extended to 64x by Result Tile in 8x1
__global__ void conv1d_2048_t8x1(const float* a, const float* w, float* __restrict__ r){
  int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
  int gid = x * 8;
  if( gid >= (WIDTH*HEIGHT)){
     return ;
  }
  float sum[8];
  float ww[16];
  float aa[8];

  //Load 8 Data 
  for(int i=0; i <8; i++)
  {
      sum[i] = 0;
      ww[i] = w[i];
      aa[i] = a[i+gid];

  }

  // SUM[0]: 8x filter
  // SUM[1]: 7x filter
  // SUM[2]: 6x filter
  // SUM[3]: 5x filter
  // SUM[4]: 4x filter
  // SUM[5]: 3x filter
  // SUM[6]: 2x filter
  // SUM[7]: 1x filter

  for(int i=0; i < 8; i++)
  {
      for(int j=0; j < (8-i);j++)
      {
         sum[i] += aa[j+i] * ww[j];
      }
       
  }

  // SUM[0-7]: 8x filter
  for(int i=8; i < 16; i+=8)
  {
     for(int s=0; s<8; s++)
     {
       aa[s] = a[gid+i+s];
       ww[s+8] = w[i+s];
     }
     for(int s =0; s < 8; s++)
     {
        int offset = 8-s;
        for(int t=0; t < 8; t++)
        {
          sum[s] += aa[t] * ww[offset + t];
        }
     }

     //shift left
     for(int s=0; s < 8; s++){
       ww[s] = ww[s+8];
     }
     
  }

  //Rest Major Loop
 //#pragma unroll  
  for(int i=16; i < 2048; i+=16)
  {
     for(int s=0; s<8; s++)
     {
       aa[s] = a[gid+i+s];
       ww[s+8] = w[i+s];
     }
     for(int s =0; s < 8; s++)
     {
        int offset = 8-s;
        for(int t=0; t < 8; t++)
        {
          sum[s] += aa[t] * ww[offset + t];
        }
     }

     //shift left
     for(int s=0; s < 8; s++){
       ww[s] = ww[s+8];
     }


     i+=8;
     for(int s=0; s<8; s++)
     {
       aa[s] = a[gid+i+s];
       ww[s+8] = w[i+s];
     }
     for(int s =0; s < 8; s++)
     {
        int offset = 8-s;
        for(int t=0; t < 8; t++)
        {
          sum[s] += aa[t] * ww[offset + t];
        }
     }

     //shift left
     for(int s=0; s < 8; s++){
       ww[s] = ww[s+4];
     }
     
  }

  //last 3
  for(int i=0; i <7; i++){
    aa[i] = a[gid+2048+i];   
  }
#if 0  
  sum[1] += aa[0] * ww[3];
  sum[2] += aa[0] * ww[2];
  sum[2] += aa[1] * ww[3];
  sum[3] += aa[0] * ww[1];
  sum[3] += aa[1] * ww[2];
  sum[3] += aa[2] * ww[3];
#else 
  for(int s=1; s < 8; s++)
  {
      for(int t=0; t <s; t++)
      {
        int offset = 8-s;
        sum[s] += aa[t] * ww[offset+t];
      }
  }
#endif 

   for(int i=0; i <8; i++){
       r[gid+i] = sum[i];
   } 
}


//BLOCK : Result  tile: 8x64
//Total waves can be extended to 64x by Result Tile in 8x1
__global__ void conv1d_2048_t8x1_unroll(const float* a, const float* w, float* __restrict__ r){
  int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;
  int gid = x * 8;
  if( gid >= (WIDTH*HEIGHT)){
     return ;
  }
  float sum[8];
  float ww[16];
  float aa[8];

  //Load 8 Data 
  for(int i=0; i <8; i++)
  {
      sum[i] = 0;
      ww[i] = w[i];
      aa[i] = a[i+gid];

  }

  // SUM[0]: 8x filter
  // SUM[1]: 7x filter
  // SUM[2]: 6x filter
  // SUM[3]: 5x filter
  // SUM[4]: 4x filter
  // SUM[5]: 3x filter
  // SUM[6]: 2x filter
  // SUM[7]: 1x filter

  for(int i=0; i < 8; i++)
  {
      for(int j=0; j < (8-i);j++)
      {
         sum[i] += aa[j+i] * ww[j];
      }
       
  }

  // SUM[0-7]: 8x filter
  for(int i=8; i < 16; i+=8)
  {
     for(int s=0; s<8; s++)
     {
       aa[s] = a[gid+i+s];
       ww[s+8] = w[i+s];
     }
     for(int s =0; s < 8; s++)
     {
        int offset = 8-s;
        for(int t=0; t < 8; t++)
        {
          sum[s] += aa[t] * ww[offset + t];
        }
     }

     //shift left
     for(int s=0; s < 8; s++){
       ww[s] = ww[s+8];
     }
     
  }

  //Rest Major Loop
 #pragma unroll  
  for(int i=16; i < 2048; i+=16)
  {
     for(int s=0; s<8; s++)
     {
       aa[s] = a[gid+i+s];
       ww[s+8] = w[i+s];
     }
     for(int s =0; s < 8; s++)
     {
        int offset = 8-s;
        for(int t=0; t < 8; t++)
        {
          sum[s] += aa[t] * ww[offset + t];
        }
     }

     //shift left
     for(int s=0; s < 8; s++){
       ww[s] = ww[s+8];
     }


     i+=8;
     for(int s=0; s<8; s++)
     {
       aa[s] = a[gid+i+s];
       ww[s+8] = w[i+s];
     }
     for(int s =0; s < 8; s++)
     {
        int offset = 8-s;
        for(int t=0; t < 8; t++)
        {
          sum[s] += aa[t] * ww[offset + t];
        }
     }

     //shift left
     for(int s=0; s < 8; s++){
       ww[s] = ww[s+4];
     }
     
  }

  //last 3
  for(int i=0; i <7; i++){
    aa[i] = a[gid+2048+i];   
  }
#if 0  
  sum[1] += aa[0] * ww[3];
  sum[2] += aa[0] * ww[2];
  sum[2] += aa[1] * ww[3];
  sum[3] += aa[0] * ww[1];
  sum[3] += aa[1] * ww[2];
  sum[3] += aa[2] * ww[3];
#else 
  for(int s=1; s < 8; s++)
  {
      for(int t=0; t <s; t++)
      {
        int offset = 8-s;
        sum[s] += aa[t] * ww[offset+t];
      }
  }
#endif 

   for(int i=0; i <8; i++){
       r[gid+i] = sum[i];
   } 
}


using namespace std;

int main() {
  
  float* hostA;
  float* hostB;
  float* hostC;

  float* deviceA;
  float* deviceB;
  float* deviceC;

  hipDeviceProp_t devProp;
  hipGetDeviceProperties(&devProp, 0);
  cout << " System minor " << devProp.minor << endl;
  cout << " System major " << devProp.major << endl;
  cout << " agent prop name " << devProp.name << endl;



  cout << "hip Device prop succeeded " << endl ;


  int i;
  int errors;

  hostA = (float*)malloc(NUM * sizeof(float));
  hostB = (float*)malloc(NUM * sizeof(float));
  hostC = (float*)malloc(NUM * sizeof(float));
  
  // initialize the input data
  for (i = 0; i < NUM; i++) {
    hostB[i] = (float)i;
    hostC[i] = (float)i*100.0f;
  }
  
  HIP_ASSERT(hipMalloc((void**)&deviceA, NUM * sizeof(float)));
  HIP_ASSERT(hipMalloc((void**)&deviceB, NUM * sizeof(float)));
  HIP_ASSERT(hipMalloc((void**)&deviceC, NUM * sizeof(float)));
  
  HIP_ASSERT(hipMemcpy(deviceB, hostB, NUM*sizeof(float), hipMemcpyHostToDevice));
  HIP_ASSERT(hipMemcpy(deviceC, hostC, NUM*sizeof(float), hipMemcpyHostToDevice));

  hipEvent_t start, stop;

	hipEventCreate(&start);
	hipEventCreate(&stop);
	float eventMs = 1.0f;

    hipLaunchKernelGGL(conv1d_2048, 
                  dim3((WIDTH*HEIGHT)/THREADS_PER_BLOCK_X),
                  dim3(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y),
                  0, 0,
                  deviceA ,deviceB ,deviceC);

		hipEventRecord(start, NULL);
	for (int i = 1; i < 64; i++)
	{
    hipLaunchKernelGGL(conv1d_2048, 
                  dim3((WIDTH*HEIGHT)/THREADS_PER_BLOCK_X),
                  dim3(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y),
                  0, 0,
                  deviceA ,deviceB ,deviceC);
	}

		hipEventRecord(stop, NULL);
		hipEventSynchronize(stop);

		hipEventElapsedTime(&eventMs, start, stop);

		//printf("elapsed time:%f\n", eventMs);
    double ips = ( double)(NUM-2048) * 2048 * 64 /1024/1024/1024;
		ips = ips / ( double)eventMs * 1000;
		printf("conv1d_2048 ==> %lf G FMAs/s\n", ips);

   {
          hipLaunchKernelGGL(conv1d_2048_opt1, 
                        dim3((WIDTH*HEIGHT)/THREADS_PER_BLOCK_X/4),
                        dim3(THREADS_PER_BLOCK_X ),
                        0, 0,
                        deviceA ,deviceB ,deviceC);

          hipEventRecord(start, NULL);
        for (int i = 1; i < 64; i++)
        {
          hipLaunchKernelGGL(conv1d_2048_opt1, 
                        dim3((WIDTH*HEIGHT)/THREADS_PER_BLOCK_X/4),
                        dim3(THREADS_PER_BLOCK_X),
                        0, 0,
                        deviceA ,deviceB ,deviceC);
        }
          hipEventRecord(stop, NULL);
          hipEventSynchronize(stop);

          hipEventElapsedTime(&eventMs, start, stop);

          //printf("elapsed time:%f\n", eventMs);
          ips =( double) (NUM-2048) * 2048 * 64 /1024/1024/1024;
          ips = ips /(double)  eventMs * 1000 ;
          printf("conv1d_2048_opt1 ==> %lf G FMAs/s, time: %f \n", ips, eventMs );
   }

{
          hipLaunchKernelGGL(conv1d_2048_opt2, 
                        dim3((WIDTH*HEIGHT)/THREADS_PER_BLOCK_X/4),
                        dim3(THREADS_PER_BLOCK_X ),
                        0, 0,
                        deviceA ,deviceB ,deviceC);

          hipEventRecord(start, NULL);
        for (int i = 1; i < 64; i++)
        {
          hipLaunchKernelGGL(conv1d_2048_opt2, 
                        dim3((WIDTH*HEIGHT)/THREADS_PER_BLOCK_X/4),
                        dim3(THREADS_PER_BLOCK_X),
                        0, 0,
                        deviceA ,deviceB ,deviceC);
        }
          hipEventRecord(stop, NULL);
          hipEventSynchronize(stop);

          hipEventElapsedTime(&eventMs, start, stop);

          //printf("elapsed time:%f\n", eventMs);
          ips =( double) (NUM-2048) * 2048 * 64 /1024/1024/1024;
          ips = ips /(double)  eventMs * 1000 ;
          printf("conv1d_2048_opt2 ==> %lf G FMAs/s, time: %f \n", ips, eventMs );
   }

{
          hipLaunchKernelGGL(conv1d_2048_opt3, 
                        dim3((WIDTH*HEIGHT)/256/4),
                        dim3(256 ),
                        0, 0,
                        deviceA ,deviceB ,deviceC);

          hipEventRecord(start, NULL);
        for (int i = 1; i < 64; i++)
        {
          hipLaunchKernelGGL(conv1d_2048_opt3, 
                        dim3((WIDTH*HEIGHT)/256/4),
                        dim3(256 ),
                        0, 0,
                        deviceA ,deviceB ,deviceC);
        }
          hipEventRecord(stop, NULL);
          hipEventSynchronize(stop);

          hipEventElapsedTime(&eventMs, start, stop);

          //printf("elapsed time:%f\n", eventMs);
          ips =( double) (NUM-2048) * 2048 * 64 /1024/1024/1024;
          ips = ips /(double)  eventMs * 1000 ;
          printf("conv1d_2048_opt2 ==> %lf G FMAs/s, time: %f \n", ips, eventMs );
   }


{
          hipLaunchKernelGGL(conv1d_2048_t8x1, 
                        dim3((WIDTH*HEIGHT)/256/8),
                        dim3(256 ),
                        0, 0,
                        deviceA ,deviceB ,deviceC);

          hipEventRecord(start, NULL);
        for (int i = 1; i < 64; i++)
        {
          hipLaunchKernelGGL(conv1d_2048_t8x1, 
                        dim3((WIDTH*HEIGHT)/256/8),
                        dim3(256 ),
                        0, 0,
                        deviceA ,deviceB ,deviceC);
        }
          hipEventRecord(stop, NULL);
          hipEventSynchronize(stop);

          hipEventElapsedTime(&eventMs, start, stop);

          //printf("elapsed time:%f\n", eventMs);
          ips =( double) (NUM-2048) * 2048 * 64 /1024/1024/1024;
          ips = ips /(double)  eventMs * 1000 ;
          printf("conv1d_2048_t8x1 ==> %lf G FMAs/s, time: %f \n", ips, eventMs );
   }

{
          hipLaunchKernelGGL(conv1d_2048_t8x1_unroll, 
                        dim3((WIDTH*HEIGHT)/256/8),
                        dim3(256 ),
                        0, 0,
                        deviceA ,deviceB ,deviceC);

          hipEventRecord(start, NULL);
        for (int i = 1; i < 64; i++)
        {
          hipLaunchKernelGGL(conv1d_2048_t8x1_unroll, 
                        dim3((WIDTH*HEIGHT)/256/8),
                        dim3(256 ),
                        0, 0,
                        deviceA ,deviceB ,deviceC);
        }
          hipEventRecord(stop, NULL);
          hipEventSynchronize(stop);

          hipEventElapsedTime(&eventMs, start, stop);

          //printf("elapsed time:%f\n", eventMs);
          ips =( double) (NUM-2048) * 2048 * 64 /1024/1024/1024;
          ips = ips /(double)  eventMs * 1000 ;
          printf("conv1d_2048_t8x1 ==> %lf G FMAs/s, time: %f \n", ips, eventMs );
   }

  HIP_ASSERT(hipMemcpy(hostA, deviceA, NUM*sizeof(float), hipMemcpyDeviceToHost));

  // verify the results

  HIP_ASSERT(hipFree(deviceA));
  HIP_ASSERT(hipFree(deviceB));
  HIP_ASSERT(hipFree(deviceC));

  free(hostA);
  free(hostB);
  free(hostC);

  //hipResetDefaultAccelerator();

  return errors;
}
