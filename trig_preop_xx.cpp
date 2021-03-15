  
/*
Copyright (c) 2015-2016 Advanced Micro Devices, Inc. All rights reserved.
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/
#include <assert.h>
#include <stdio.h>
#include <algorithm>
#include <stdlib.h>
#include<iostream>
#include "hip/hip_runtime.h"


#define HIP_ASSERT(x) (assert((x)==hipSuccess))


#define NUM       (2048)


__global__ void 
trig_preop_xx(double* __restrict__ a, const double* __restrict__ b){
    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;     
    
    double data =  b[x];
    

      a[ x +  31 *2048] = data;
    
    
    for(int i=0; i<31; i++){
      int seg = i;
      double result = 0;
      asm volatile("v_trig_preop_f64 %0, %1, %2":: "v"(result), "v"(data), "v"(seg));
      asm volatile("s_waitcnt 0\n");    
      a[ x + i * 2048] = result;
    }
    
} 

using namespace std;

int main() {
  
  double* hostA;
  double* hostB;
  double* hostC;

  double* deviceA;
  double* deviceB;
  double* deviceC;

  hipDeviceProp_t devProp;
  hipGetDeviceProperties(&devProp, 0);
  cout << " System minor " << devProp.minor << endl;
  cout << " System major " << devProp.major << endl;
  cout << " agent prop name " << devProp.name << endl;



  cout << "hip Device prop succeeded " << endl ;


  long i;
  int errors;

  hostA = (double*)malloc(NUM * 32 * sizeof(double));
  hostB = (double*)malloc(NUM * sizeof(double));
  
  // initialize the input data
  double seed = 3.15;
  for (i = 0; i < NUM; i++) {
    hostB[i] = seed;
    hostA[i] = 0.0;
    seed *= 1.4135;
  }
  
  HIP_ASSERT(hipMalloc((void**)&deviceA, NUM *32* sizeof(double)));
  HIP_ASSERT(hipMalloc((void**)&deviceB, NUM * sizeof(double)));
  
  HIP_ASSERT(hipMemcpy(deviceB, hostB, NUM*sizeof(double), hipMemcpyHostToDevice));  

   {
        //hipEventSynchronize(NULL);
        hipEvent_t start, stop; 
        hipLaunchKernelGGL(trig_preop_xx, 
                    dim3(32),
                    dim3(64),
                    0, 0,
                    deviceA ,deviceB );

      //hipEventSynchronize(NULL);
      HIP_ASSERT(hipMemcpy(hostA, deviceA, NUM*32*sizeof(double), hipMemcpyDeviceToHost));
      //hipEventSynchronize(NULL);

      for( i = 0; i < (NUM * 32) ; i+=16){
        for(int j = 0; j < 16; j++)        {
            
            printf("%g ", hostA[i+j]);
        }
        printf("\n");
      }
   }

  HIP_ASSERT(hipFree(deviceA));
  HIP_ASSERT(hipFree(deviceB));

  free(hostA);
  free(hostB);

  //hipResetDefaultAccelerator();

  return errors;
}
