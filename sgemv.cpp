#include <assert.h>
#include <stdio.h>
#include <algorithm>
#include <stdlib.h>
#include<iostream>
#include "hip/hip_runtime.h"


#define HIP_ASSERT(x) (assert((x)==hipSuccess))


#define M    8192
#define N    (4096)
#define NN   (4096+32)

#define NUM       (M*NN)

#define THREADS_PER_BLOCK_X  64
#define THREADS_PER_BLOCK_Y  1
#define THREADS_PER_BLOCK_Z  1


//Marix  A in normal format:  M*N
//Matrix B in N *1  or 1 *N
//Matrix C will be M * 1
//WorkGroup_SIZE  64
//WK Group TIlE： result C in 16 * 1 
//WK Grouop       Matrix A in 16 *N,   Matrix B in N * 1 
// 64 threads compute same result
// Thread Tile  per iteration : Result C in 16 * 1 , Matrix A 16 * 4, Matrix B 4 * 1 
// Thread 0 fetch A base : 0;
// Thread 1 fetch A base： 4; 
// Thread 2 fetch A base: 8,... 
// Thread 0 fetch B base : 0;
// Thread 1 fetch B base : 4;
// Thread 2 fetch B base : 8;...


#define  BLOCK_C_TILE_X 16
#define THREAD_LOAD_NUM  4 
#define BLOCK_LOAD_NUM  (THREAD_LOAD_NUM * THREADS_PER_BLOCK_X)
__global__ void sgemv_16x1(const float* a, const float* b, float* __restrict__ c, const int m, const int n, const int lda ){
    int wk_tile_m =  hipBlockIdx_x * BLOCK_C_TILE_X ;
    
    int offset = hipThreadIdx_x * THREAD_LOAD_NUM;

    //NO Preload     
    float sum[BLOCK_C_TILE_X];
    float* a_ptr = (float*)a  + (wk_tile_m * lda);
    for(int i=0; i < BLOCK_C_TILE_X; i++)
        sum[i] = 0;
    for(int i =0; i < n;  i+= BLOCK_LOAD_NUM ) {
        //LOAD B
        float b_data[THREAD_LOAD_NUM]; 

        //Matrix B: 4x1 per thread
        for(int j=0; j < THREAD_LOAD_NUM; j++)
            b_data[j] = b[offset+j]; 


        //Matrix A: 16X4 per thread,
#pragma unroll        
        for(int j=0; j < BLOCK_C_TILE_X; j++)
        {
             //Load A
            float a_data[THREAD_LOAD_NUM]; 

            for(int k=0; k < THREAD_LOAD_NUM; k++ ){
                a_data[k]  = a_ptr[j * lda + offset + k];

                //SUM A
                sum[j] += a_data[k] * b_data[k];
            }
        }      

        //Move offset 
        offset +=  BLOCK_LOAD_NUM;
    }

    //Reduction
    __shared__ float s_sum[BLOCK_C_TILE_X * THREADS_PER_BLOCK_X];

    //Store into LDS first 
    for(int i= 0; i < BLOCK_C_TILE_X; i++){
         s_sum[ i * THREADS_PER_BLOCK_X + hipThreadIdx_x ] = sum[i];
    }    

    
    //64X  threads Do reduction  together
    //every time reduce 8 value
    //s_sum[0] has 64x 
    //Thread 0-7 recude sum[0]
    //Thread 8-15 reduce SUM[1]
    //...
    //Thread 56,63 reduce SUM[7] 
  
    int s_read_offset  = (hipThreadIdx_x >> 3) * THREADS_PER_BLOCK_X + (hipThreadIdx_x&0x7) * 8;
    int s_write_offset = hipThreadIdx_x;

    __syncthreads();
    //Reducce every 8x value into 1 thread, 1 C needs 8 threads. total 64 threads.
    s_sum[s_write_offset]  =  s_sum[s_read_offset + 0] +
                              s_sum[s_read_offset + 1] +  
                              s_sum[s_read_offset + 2] +  
                              s_sum[s_read_offset + 3] +  
                              s_sum[s_read_offset + 4] +  
                              s_sum[s_read_offset + 5] +  
                              s_sum[s_read_offset + 6] +  
                              s_sum[s_read_offset + 7]
                              ;  
    //Reduce next 8x value
    s_read_offset   += THREADS_PER_BLOCK_X * 8;
    s_write_offset  += THREADS_PER_BLOCK_X;
    s_sum[s_write_offset]  =  s_sum[s_read_offset + 0] +
                              s_sum[s_read_offset + 1] +  
                              s_sum[s_read_offset + 2] +  
                              s_sum[s_read_offset + 3] +  
                              s_sum[s_read_offset + 4] +  
                              s_sum[s_read_offset + 5] +  
                              s_sum[s_read_offset + 6] +  
                              s_sum[s_read_offset + 7]
                              ;  
    
    __syncthreads();
    //first 16 threads reduces 8x data 
    if( hipThreadIdx_x < 16){
        s_read_offset = hipThreadIdx_x  * 8;

        sum[0] = s_sum[s_read_offset + 0] + 
                 s_sum[s_read_offset + 1] +  
                 s_sum[s_read_offset + 2] + 
                 s_sum[s_read_offset + 3] + 
                 s_sum[s_read_offset + 4] + 
                 s_sum[s_read_offset + 5] + 
                 s_sum[s_read_offset + 6] + 
                 s_sum[s_read_offset + 7]  
              ;
    }

    //Store into memory

    if( hipThreadIdx_x < 16){
      c[wk_tile_m + hipThreadIdx_x] = sum[0];
    }
}



#undef BLOCK_C_TILE_X
#undef THREAD_LOAD_NUM
#undef  BLOCK_LOAD_NUM
#define  BLOCK_C_TILE_X 8
#define THREAD_LOAD_NUM  4 
#define BLOCK_LOAD_NUM  (THREAD_LOAD_NUM * THREADS_PER_BLOCK_X)
__global__ void sgemv_8x1(const float* a, const float* b, float* __restrict__ c, const int m, const int n, const int lda ){
    int wk_tile_m =  hipBlockIdx_x * BLOCK_C_TILE_X ;
    
    int offset = hipThreadIdx_x * THREAD_LOAD_NUM;

    //NO Preload     
    float sum[BLOCK_C_TILE_X];
    float* a_ptr = (float*)a  + (wk_tile_m * lda);
    for(int i=0; i < BLOCK_C_TILE_X; i++)
        sum[i] = 0;
#if 0        
    for(int i =0; i < n;  i+= BLOCK_LOAD_NUM  ) {
        //LOAD B
        float b_data[THREAD_LOAD_NUM]; 

        //Matrix B: 4x1 per thread
        for(int j=0; j < THREAD_LOAD_NUM; j++)
            b_data[j] = b[offset+j]; 


        //Matrix A: 16X4 per thread,
#pragma unroll        
        for(int j=0; j < BLOCK_C_TILE_X; j++)
        {
             //Load A
            float a_data[THREAD_LOAD_NUM]; 

            for(int k=0; k < THREAD_LOAD_NUM; k++ ){
                a_data[k]  = a_ptr[j * lda + offset + k];

                //SUM A
                sum[j] += a_data[k] * b_data[k];
            }
        }      

        //Move offset 
        offset +=  BLOCK_LOAD_NUM;
    }
#else 
    //4x FMA per M per Thread 
    //32 FMA per 8M per threads
    //256 FMA per M per workgroup
    for(int i =0; i < n;  i+= BLOCK_LOAD_NUM  ) {
        //LOAD B
        float b_data[THREAD_LOAD_NUM]; 

        //Matrix B: 4x1 per thread
        for(int j=0; j < THREAD_LOAD_NUM; j++)
            b_data[j] = b[offset+j]; 

        //Matrix A: 16X4 per thread,
#pragma unroll        
        for(int j=0; j < BLOCK_C_TILE_X; j++)
        {
             //Load A
            float a_data[THREAD_LOAD_NUM]; 

            for(int k=0; k < THREAD_LOAD_NUM; k++ ){
                a_data[k]  = a_ptr[j * lda + offset + k];

                //SUM A
                sum[j] += a_data[k] * b_data[k];
            }
        }      

        //Move offset 
        offset +=  BLOCK_LOAD_NUM;


#if 0        
        //Matrix B: 4x1 per thread
        for(int j=0; j < THREAD_LOAD_NUM; j++)
            b_data[j] = b[offset+j]; 


        //Matrix A: 16X4 per thread,
#pragma unroll        
        for(int j=0; j < BLOCK_C_TILE_X; j++)
        {
             //Load A
            float a_data[THREAD_LOAD_NUM]; 

            for(int k=0; k < THREAD_LOAD_NUM; k++ ){
                a_data[k]  = a_ptr[j * lda + offset + k];

                //SUM A
                sum[j] += a_data[k] * b_data[k];
            }
        }
        offset +=  BLOCK_LOAD_NUM;      
#endif        
    }
#endif    

    //Reduction
    __shared__ float s_sum[BLOCK_C_TILE_X * THREADS_PER_BLOCK_X];

    //Store into LDS first 
    for(int i= 0; i < BLOCK_C_TILE_X; i++){
         s_sum[ i * THREADS_PER_BLOCK_X + hipThreadIdx_x ] = sum[i];
    }    

    
    //64X  threads Do reduction  together
    //every time reduce 8 value
    //Thread 0-7  reduce SUM[0] 
    //Thread 8-15 reduce SUM[1] 
    //...
    //Thread 56,63 reduce SUM[7]
  
    int s_read_offset  = (hipThreadIdx_x >> 3) * THREADS_PER_BLOCK_X + (hipThreadIdx_x&0x7) * 8;
    int s_write_offset = hipThreadIdx_x;

    __syncthreads();
    //__syncthreads 8 SUM by 64 threads 
    s_sum[s_write_offset]  =  s_sum[s_read_offset + 0] +
                              s_sum[s_read_offset + 1] +  
                              s_sum[s_read_offset + 2] +  
                              s_sum[s_read_offset + 3] +  
                              s_sum[s_read_offset + 4] +  
                              s_sum[s_read_offset + 5] +  
                              s_sum[s_read_offset + 6] +  
                              s_sum[s_read_offset + 7]
                              ;  
    
    __syncthreads();
    
    //first 8 threads reduces 8x data 
    if( hipThreadIdx_x < 8){
        s_read_offset = hipThreadIdx_x  * 8;

        sum[0] = s_sum[s_read_offset + 0] + 
                 s_sum[s_read_offset + 1] +  
                 s_sum[s_read_offset + 2] + 
                 s_sum[s_read_offset + 3] + 
                 s_sum[s_read_offset + 4] + 
                 s_sum[s_read_offset + 5] + 
                 s_sum[s_read_offset + 6] + 
                 s_sum[s_read_offset + 7]  
              ;
    }

    //Store into memory

    if( hipThreadIdx_x < 8){
      c[wk_tile_m + hipThreadIdx_x] = sum[0];
    }
}


#undef BLOCK_C_TILE_X
#undef THREAD_LOAD_NUM
#undef  BLOCK_LOAD_NUM
#define  BLOCK_C_TILE_X  4
#define THREAD_LOAD_NUM  4 
#define BLOCK_LOAD_NUM  (THREAD_LOAD_NUM * THREADS_PER_BLOCK_X)
__global__ void sgemv_4x1(const float* a, const float* b, float* __restrict__ c, const int m, const int n, const int lda ){
    int wk_tile_m =  hipBlockIdx_x * BLOCK_C_TILE_X ;
    
    int offset = hipThreadIdx_x * THREAD_LOAD_NUM;

    //NO Preload     
    float sum[BLOCK_C_TILE_X];
    float* a_ptr = (float*)a  + (wk_tile_m * lda);
    for(int i=0; i < BLOCK_C_TILE_X; i++)
        sum[i] = 0;
#if 0        
    for(int i =0; i < n;  i+= BLOCK_LOAD_NUM  ) {
        //LOAD B
        float b_data[THREAD_LOAD_NUM]; 

        //Matrix B: 4x1 per thread
        for(int j=0; j < THREAD_LOAD_NUM; j++)
            b_data[j] = b[offset+j]; 


        //Matrix A: 16X4 per thread,
#pragma unroll        
        for(int j=0; j < BLOCK_C_TILE_X; j++)
        {
             //Load A
            float a_data[THREAD_LOAD_NUM]; 

            for(int k=0; k < THREAD_LOAD_NUM; k++ ){
                a_data[k]  = a_ptr[j * lda + offset + k];

                //SUM A
                sum[j] += a_data[k] * b_data[k];
            }
        }      

        //Move offset 
        offset +=  BLOCK_LOAD_NUM;
    }
#else 
    //4x FMA per M per Thread 
    //32 FMA per 8M per threads
    //256 FMA per M per workgroup
    for(int i =0; i < n;  i+= BLOCK_LOAD_NUM  ) {
        //LOAD B
        float b_data[THREAD_LOAD_NUM]; 

        //Matrix B: 4x1 per thread
        for(int j=0; j < THREAD_LOAD_NUM; j++)
            b_data[j] = b[offset+j]; 

        //Matrix A: 16X4 per thread,
#pragma unroll        
        for(int j=0; j < BLOCK_C_TILE_X; j++)
        {
             //Load A
            float a_data[THREAD_LOAD_NUM]; 

            for(int k=0; k < THREAD_LOAD_NUM; k++ ){
                a_data[k]  = a_ptr[j * lda + offset + k];

                //SUM A
                sum[j] += a_data[k] * b_data[k];
            }
        }      

        //Move offset 
        offset +=  BLOCK_LOAD_NUM;


#if 0        
        //Matrix B: 4x1 per thread
        for(int j=0; j < THREAD_LOAD_NUM; j++)
            b_data[j] = b[offset+j]; 


        //Matrix A: 16X4 per thread,
#pragma unroll        
        for(int j=0; j < BLOCK_C_TILE_X; j++)
        {
             //Load A
            float a_data[THREAD_LOAD_NUM]; 

            for(int k=0; k < THREAD_LOAD_NUM; k++ ){
                a_data[k]  = a_ptr[j * lda + offset + k];

                //SUM A
                sum[j] += a_data[k] * b_data[k];
            }
        }
        offset +=  BLOCK_LOAD_NUM;      
#endif        
    }
#endif    

    //Reduction
    __shared__ float s_sum[BLOCK_C_TILE_X * THREADS_PER_BLOCK_X];

    //Store into LDS first 
    for(int i= 0; i < BLOCK_C_TILE_X; i++){
         s_sum[ i * THREADS_PER_BLOCK_X + hipThreadIdx_x ] = sum[i];
    }    

    
    //64X  threads Do reduction  together
    //every time reduce 8 value
    //Thread 0-7  reduce SUM[0] 
    //Thread 8-15 reduce SUM[1] 
    //Thread 15-23  reduce SUM[0] 
    //Thread 24-31 reduce SUM[1] 
  
    int s_read_offset  = (hipThreadIdx_x >> 3) * THREADS_PER_BLOCK_X + (hipThreadIdx_x&0x7) * 8;
    int s_write_offset = hipThreadIdx_x;

    __syncthreads();
    //__syncthreads 8 SUM by 64 threads 
    if( hipThreadIdx_x < 32) {
    s_sum[s_write_offset]  =  s_sum[s_read_offset + 0] +
                              s_sum[s_read_offset + 1] +  
                              s_sum[s_read_offset + 2] +  
                              s_sum[s_read_offset + 3] +  
                              s_sum[s_read_offset + 4] +  
                              s_sum[s_read_offset + 5] +  
                              s_sum[s_read_offset + 6] +  
                              s_sum[s_read_offset + 7]
                              ;  
    }
    
    __syncthreads();
    
    //first 8 threads reduces 8x data 
    if( hipThreadIdx_x < BLOCK_C_TILE_X){
        s_read_offset = hipThreadIdx_x  * 8;
        sum[0] = s_sum[s_read_offset + 0] + 
                 s_sum[s_read_offset + 1] +  
                 s_sum[s_read_offset + 2] + 
                 s_sum[s_read_offset + 3] + 
                 s_sum[s_read_offset + 4] + 
                 s_sum[s_read_offset + 5] + 
                 s_sum[s_read_offset + 6] + 
                 s_sum[s_read_offset + 7]  
              ;
    }

    //Store into memory

    if( hipThreadIdx_x < BLOCK_C_TILE_X){
      c[wk_tile_m + hipThreadIdx_x] = sum[0];
    }
}

#undef BLOCK_C_TILE_X
#undef THREAD_LOAD_NUM
#undef  BLOCK_LOAD_NUM
#define  BLOCK_C_TILE_X  2
#define THREAD_LOAD_NUM  4 
#define BLOCK_LOAD_NUM  (THREAD_LOAD_NUM * THREADS_PER_BLOCK_X)
__global__ void sgemv_2x1(const float* a, const float* b, float* __restrict__ c, const int m, const int n, const int lda ){
    int wk_tile_m =  hipBlockIdx_x * BLOCK_C_TILE_X ;
    
    int offset = hipThreadIdx_x * THREAD_LOAD_NUM;

    //NO Preload     
    float sum[BLOCK_C_TILE_X];
    float* a_ptr = (float*)a  + (wk_tile_m * lda);
    for(int i=0; i < BLOCK_C_TILE_X; i++)
        sum[i] = 0;
#if 0        
    for(int i =0; i < n;  i+= BLOCK_LOAD_NUM  ) {
        //LOAD B
        float b_data[THREAD_LOAD_NUM]; 

        //Matrix B: 4x1 per thread
        for(int j=0; j < THREAD_LOAD_NUM; j++)
            b_data[j] = b[offset+j]; 


        //Matrix A: 16X4 per thread,
#pragma unroll        
        for(int j=0; j < BLOCK_C_TILE_X; j++)
        {
             //Load A
            float a_data[THREAD_LOAD_NUM]; 

            for(int k=0; k < THREAD_LOAD_NUM; k++ ){
                a_data[k]  = a_ptr[j * lda + offset + k];

                //SUM A
                sum[j] += a_data[k] * b_data[k];
            }
        }      

        //Move offset 
        offset +=  BLOCK_LOAD_NUM;
    }
#else 
    //4x FMA per M per Thread 
    //32 FMA per 8M per threads
    //256 FMA per M per workgroup
    for(int i =0; i < n;  i+= BLOCK_LOAD_NUM  ) {
        //LOAD B
        float b_data[THREAD_LOAD_NUM]; 

        //Matrix B: 4x1 per thread
        for(int j=0; j < THREAD_LOAD_NUM; j++)
            b_data[j] = b[offset+j]; 

        //Matrix A: 16X4 per thread,
#pragma unroll        
        for(int j=0; j < BLOCK_C_TILE_X; j++)
        {
             //Load A
            float a_data[THREAD_LOAD_NUM]; 

            for(int k=0; k < THREAD_LOAD_NUM; k++ ){
                a_data[k]  = a_ptr[j * lda + offset + k];

                //SUM A
                sum[j] += a_data[k] * b_data[k];
            }
        }      

        //Move offset 
        offset +=  BLOCK_LOAD_NUM;


#if 0        
        //Matrix B: 4x1 per thread
        for(int j=0; j < THREAD_LOAD_NUM; j++)
            b_data[j] = b[offset+j]; 


        //Matrix A: 16X4 per thread,
#pragma unroll        
        for(int j=0; j < BLOCK_C_TILE_X; j++)
        {
             //Load A
            float a_data[THREAD_LOAD_NUM]; 

            for(int k=0; k < THREAD_LOAD_NUM; k++ ){
                a_data[k]  = a_ptr[j * lda + offset + k];

                //SUM A
                sum[j] += a_data[k] * b_data[k];
            }
        }
        offset +=  BLOCK_LOAD_NUM;      
#endif        
    }
#endif    

    //Reduction
    __shared__ float s_sum[BLOCK_C_TILE_X * THREADS_PER_BLOCK_X];

    //Store into LDS first 
    for(int i= 0; i < BLOCK_C_TILE_X; i++){
         s_sum[ i * THREADS_PER_BLOCK_X + hipThreadIdx_x ] = sum[i];
    }    

    
    //64X  threads Do reduction  together
    //every time reduce 8 value
    //Thread 0-7  reduce SUM[0] 
    //Thread 8-15 reduce SUM[1] 
  
    int s_read_offset  = (hipThreadIdx_x >> 3) * THREADS_PER_BLOCK_X + (hipThreadIdx_x&0x7) * 8;
    int s_write_offset = hipThreadIdx_x;

    __syncthreads();
    //__syncthreads 8 SUM by 64 threads 
    if( hipThreadIdx_x < 16) {
    s_sum[s_write_offset]  =  s_sum[s_read_offset + 0] +
                              s_sum[s_read_offset + 1] +  
                              s_sum[s_read_offset + 2] +  
                              s_sum[s_read_offset + 3] +  
                              s_sum[s_read_offset + 4] +  
                              s_sum[s_read_offset + 5] +  
                              s_sum[s_read_offset + 6] +  
                              s_sum[s_read_offset + 7]
                              ;  
    }
    
    __syncthreads();
    
    //first 8 threads reduces 8x data 
    if( hipThreadIdx_x < BLOCK_C_TILE_X){
        s_read_offset = hipThreadIdx_x  * 8;

        sum[0] = s_sum[s_read_offset + 0] + 
                 s_sum[s_read_offset + 1] +  
                 s_sum[s_read_offset + 2] + 
                 s_sum[s_read_offset + 3] + 
                 s_sum[s_read_offset + 4] + 
                 s_sum[s_read_offset + 5] + 
                 s_sum[s_read_offset + 6] + 
                 s_sum[s_read_offset + 7]  
              ;
    }

    //Store into memory

    if( hipThreadIdx_x < BLOCK_C_TILE_X){
      c[wk_tile_m + hipThreadIdx_x] = sum[0];
    }
}

__global__ void sgemv_direct_64x1_t1x1(const float* a, const float* b, float* __restrict__ c, const int m, const int n, const int lda ){    
    int gid = hipBlockIdx_x * 64 + hipThreadIdx_x;
    int thread_offset =  gid *lda;
    //NO Preload     
    float sum = 0;
    __shared__ float b_shared[128];
    for(int i=0;  i < n; i+=128){

       //Load 128 B into Shared memory since compiler has bug to use scalar memory for b
       b_shared[hipBlockIdx_x * 2 + 0]  =  b[i + hipBlockIdx_x * 2 + 0];
       b_shared[hipBlockIdx_x * 2 + 1]  =  b[i + hipBlockIdx_x * 2 + 1];        

#pragma unroll      
      for(int j=0; j < 128; j++){          
          sum += a[thread_offset + i + j ] * b_shared[j];
      }
    }
    c[gid] = sum;
}


//WK_SIZE: 64 
//8 threads compute 1 result
//Each Thread produce 8x1 result 
//8 threads process 8*32*1  data
//each Loop: 
//Thread 0:  8* 4 * 1 : offset N: 0 
//Thread 1:  8* 4 * 1   offset N: 4
//Thread 3:  8* 4 * 1   offet  N: 8
//Thread 4:  8* 4 * 1   offset N: 12
///... 
//Thread 7:  8* 4 * 1    offset N 28


__global__ void sgemv_direct_64x1_t8x1(const float* a, const float* b, float* __restrict__ c, const int m, const int n, const int lda ){    
    int gid = hipBlockIdx_x * 64 + ((hipThreadIdx_x >> 3) * 8);
    int thread_offset =  gid *lda + (hipThreadIdx_x & 0x7) * 4;
    //NO Preload     
    float sum[8];

    __shared__ float b_shared[32];


    for(int i=0; i < 8; i++){
        sum[i] = 0;
    }
    
    for(int i=0;  i < n; i+=32){

       //Load 128 B into Shared memory since compiler has bug to use scalar memory for b
       if(hipBlockIdx_x < 0)
              b_shared[hipBlockIdx_x]  =  b[i + hipBlockIdx_x] ;       

#pragma unroll      
      //FMA 8x4x1 
      for(int j=0; j < 8; j++){          
          sum[j] += a[thread_offset + j * lda + i + 0 ] * b_shared[ (hipThreadIdx_x & 0x7) * 4 + 0 ];
          sum[j] += a[thread_offset + j * lda + i + 1 ] * b_shared[ (hipThreadIdx_x & 0x7) * 4 + 1 ];
          sum[j] += a[thread_offset + j * lda + i + 2 ] * b_shared[ (hipThreadIdx_x & 0x7) * 4 + 2 ];
          sum[j] += a[thread_offset + j * lda + i + 3 ] * b_shared[ (hipThreadIdx_x & 0x7) * 4 + 3 ];
      }      
    }

    //reduction 
    __shared__ float sum_shared[9*64];  //stride 9 to reduce bank conflicts
    for(int i = 0; i < 8; i++){
        int c_id   = i + (hipThreadIdx_x >> 3) * 8;
        int offset = c_id * 9 + (hipThreadIdx_x & 0x7);

        sum_shared[ offset] = sum[i];
    }
    
    __syncthreads();

    //Reduction by 64 thrads so that each thread read 8x data
    sum[0] = 0;
    for(int i = 0; i < 8; i++){
      sum[0] += sum_shared[hipThreadIdx_x * 9 + i];  
    } 
    int gid2 = hipBlockIdx_x * 64 + hipThreadIdx_x;
    c[gid2] = sum[0];
}


//WK_SIZE: 64 
//8 threads compute 1 result
//Each Thread produce 8x1 result 
//32 threads process 8 * 128 * 1 
//each Loop: 
//Thread 0:  8* 4 * 1 : offset N: 0 
//Thread 1:  8* 4 * 1   offset N: 4
//Thread 3:  8* 4 * 1   offet  N: 8
//Thread 4:  8* 4 * 1   offset N: 12
///... 
//Thread 31:  8* 4 * 1  offset N： 124 
//128x of Matrix B per loop

__global__ void sgemv_direct_16x1_t8x1(const float* a, const float* b, float* __restrict__ c, const int m, const int n, const int lda ){    
    int gid = hipBlockIdx_x * 16 + ((hipThreadIdx_x >> 5) * 8);
    int thread_offset =  gid *lda + (hipThreadIdx_x &0x1f) * 4 ;
    //NO Preload     
    float sum[8];

    __shared__ float b_shared[64 * 2 ]; 
    for(int i=0; i < 8; i++){
        sum[i] = 0;
    }
    
    for(int i=0;  i < n; i+=128){

       //Load 128 B into Shared memory since compiler has bug to use scalar memory for b       
       b_shared[hipBlockIdx_x*2 +0 ]  =  b[i + hipBlockIdx_x * 2 + 0] ;       
       b_shared[hipBlockIdx_x*2 +1 ]  =  b[i + hipBlockIdx_x * 2 + 1] ;

#pragma unroll      
      //FMA 8x4x1 
      for(int j=0; j < 8; j++){          
          sum[j] += a[thread_offset + j * lda + i + 0 ] * b_shared[ (hipThreadIdx_x & 0x1f) * 4 + 0 ];
          sum[j] += a[thread_offset + j * lda + i + 1 ] * b_shared[ (hipThreadIdx_x & 0x1f) * 4 + 1 ];
          sum[j] += a[thread_offset + j * lda + i + 2 ] * b_shared[ (hipThreadIdx_x & 0x1f) * 4 + 2 ];
          sum[j] += a[thread_offset + j * lda + i + 3 ] * b_shared[ (hipThreadIdx_x & 0x1f) * 4 + 3 ];
      }
    }

    //reduction 32 thrads to 1 
    __shared__ float sum_shared[16 * 33];  //stride  33 
    for(int i = 0; i < 8; i++){
        int c_id   = i + (hipThreadIdx_x >> 5) * 8;
        int offset = c_id * 33 + (hipThreadIdx_x & 0x1f);
        sum_shared[offset] = sum[i];
    }    
    __syncthreads();

    //round1  64 * 8  
    //Thread 0: C0[0,7]
    //Thread 1: C0[8,15]
    //Thread 2: C0[16,23]
    //Thread 3: C0[24,31]

    sum[0] = 0;
    for(int i = 0; i < 8; i++){      
      int c_id = (hipThreadIdx_x >> 2);
      sum[0] += sum_shared[c_id * 9 + i];  
    } 
    sum_shared[hipThreadIdx_x] = sum[0];

    __syncthreads();
    sum[0] = 0;
    if(hipThreadIdx_x < 16){
      for(int i=0; i < 4; i++){      
        sum[0] += sum_shared[hipThreadIdx_x * 4 + i ];
      }
    }

    if(hipThreadIdx_x < 16){
      int gid2 = hipBlockIdx_x * 16 + hipThreadIdx_x;
      c[gid2] = sum[0];
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
  hostB = (float*)malloc(N * sizeof(float));
  hostC = (float*)malloc(M * sizeof(float));
  
  // initialize the input data
  for (i = 0; i < NUM; i++) {
    hostA[i] = (float)sinf(i);
  }
  for (i = 0; i < N; i++) {
    hostB[i] = (float)cosf(i);
  }

  
  HIP_ASSERT(hipMalloc((void**)&deviceA, NUM * sizeof(float)));
  HIP_ASSERT(hipMalloc((void**)&deviceB, N * sizeof(float)));
  HIP_ASSERT(hipMalloc((void**)&deviceC, M * sizeof(float)));
  
  HIP_ASSERT(hipMemcpy(deviceA, hostA, NUM*sizeof(float), hipMemcpyHostToDevice));
  HIP_ASSERT(hipMemcpy(deviceB, hostB, N*sizeof(float), hipMemcpyHostToDevice));

  hipEvent_t start, stop;

	hipEventCreate(&start);
	hipEventCreate(&stop);
	float eventMs = 0.0f;
  
   
   for(int mm=512; mm <=M; mm+=256)
   {
          hipLaunchKernelGGL(sgemv_16x1, 
                        dim3(M/16 ),
                        dim3(THREADS_PER_BLOCK_X),
                        0, 0,
                        deviceA ,deviceB ,deviceC, M, N, NN);

          hipEventRecord(start, NULL);
        for (int i = 0; i < 10; i++)
        {
          hipLaunchKernelGGL(sgemv_16x1, 
                        dim3(M/16 ),
                        dim3(THREADS_PER_BLOCK_X),
                        0, 0,
                        deviceA ,deviceB ,deviceC, M, N, NN);
        }

          hipEventRecord(stop, NULL);
          hipEventSynchronize(stop);

          hipEventElapsedTime(&eventMs, start, stop);
   
		      //printf("elapsed time:%f\n", eventMs);
          double total_bytes = ( double)(mm)* (double)N + double(mm) + double(N);          
          total_bytes = total_bytes * sizeof(float) /1024/1024/1024;
		      double gbps = total_bytes/eventMs * 1000 * 10;
		      printf("sgemv_16x1 [mm=%d] ==> %lf G Bytes/s, ms: %f\n", mm, gbps, eventMs);
   }

   for(int mm=512; mm <=M; mm+=256)
   {
          hipLaunchKernelGGL(sgemv_8x1, 
                        dim3(mm/8 ),
                        dim3(THREADS_PER_BLOCK_X),
                        0, 0,
                        deviceA ,deviceB ,deviceC, M, N, NN);

          hipEventRecord(start, NULL);
        for (int i = 0; i < 10; i++)
        {
          hipLaunchKernelGGL(sgemv_8x1, 
                        dim3(mm/8 ),
                        dim3(THREADS_PER_BLOCK_X),
                        0, 0,
                        deviceA ,deviceB ,deviceC, M, N, NN);
        }

          hipEventRecord(stop, NULL);
          hipEventSynchronize(stop);

          hipEventElapsedTime(&eventMs, start, stop);
   
		      //printf("elapsed time:%f\n", eventMs);
          double total_bytes = ( double)(mm)* (double)N + double(mm) + double(N);          
          total_bytes = total_bytes * sizeof(float) /1024/1024/1024;
		      double gbps = total_bytes/eventMs * 1000 * 10;
		      printf("sgemv_8x1 [mm=%d] ==> %lf G Bytes/s, ms: %f\n", mm, gbps, eventMs);
   }


   for(int mm=512; mm <=M; mm+=256)
   {
          hipLaunchKernelGGL(sgemv_4x1, 
                        dim3(mm/4 ),
                        dim3(THREADS_PER_BLOCK_X),
                        0, 0,
                        deviceA ,deviceB ,deviceC, M, N, NN);

          hipEventRecord(start, NULL);
        for (int i = 0; i < 10; i++)
        {
          hipLaunchKernelGGL(sgemv_4x1, 
                        dim3(mm/4 ),
                        dim3(THREADS_PER_BLOCK_X),
                        0, 0,
                        deviceA ,deviceB ,deviceC, M, N, NN);
        }

          hipEventRecord(stop, NULL);
          hipEventSynchronize(stop);

          hipEventElapsedTime(&eventMs, start, stop);
   
		      //printf("elapsed time:%f\n", eventMs);
          double total_bytes = ( double)(mm)* (double)N + double(mm) + double(N);          
          total_bytes = total_bytes * sizeof(float) /1024/1024/1024;
		      double gbps = total_bytes/eventMs * 1000 * 10;
		      printf("sgemv_4x1 [mm=%d] ==> %lf G Bytes/s, ms: %f\n", mm, gbps, eventMs);
   }

   for(int mm=512; mm <=M; mm+=256)
   {
          hipLaunchKernelGGL(sgemv_2x1, 
                        dim3(mm/2 ),
                        dim3(THREADS_PER_BLOCK_X),
                        0, 0,
                        deviceA ,deviceB ,deviceC, M, N, NN);

          hipEventRecord(start, NULL);
        for (int i = 0; i < 10; i++)
        {
          hipLaunchKernelGGL(sgemv_2x1, 
                        dim3(mm/2 ),
                        dim3(THREADS_PER_BLOCK_X),
                        0, 0,
                        deviceA ,deviceB ,deviceC, M, N, NN);
        }

          hipEventRecord(stop, NULL);
          hipEventSynchronize(stop);

          hipEventElapsedTime(&eventMs, start, stop);
   
		      //printf("elapsed time:%f\n", eventMs);
          double total_bytes = ( double)(mm)* (double)N + double(mm) + double(N);          
          total_bytes = total_bytes * sizeof(float) /1024/1024/1024;
		      double gbps = total_bytes/eventMs * 1000 * 10;
		      printf("sgemv_2x1 [mm=%d] ==> %lf G Bytes/s, ms: %f\n", mm, gbps, eventMs);
   }

  //exit(0);
   for(int mm=1024; mm <=M; mm+=256){
        hipLaunchKernelGGL(sgemv_direct_64x1_t1x1, 
                        dim3(mm/64 ),
                        dim3(64),
                        0, 0,
                        deviceA ,deviceB ,deviceC, M, N, N);

        hipEventRecord(start, NULL);
        for (int i = 0; i < 10; i++){
          hipLaunchKernelGGL(sgemv_direct_64x1_t1x1, 
                        dim3(mm/64),
                        dim3(64),
                        0, 0,
                        deviceA ,deviceB ,deviceC, M, N, NN);

        }

        hipEventRecord(stop, NULL);
        hipEventSynchronize(stop);

        hipEventElapsedTime(&eventMs, start, stop);
  
        //printf("elapsed time:%f\n", eventMs);
        float total_bytes = ( double)(mm)* (double)N + double(N) + double(mm);          
        total_bytes = total_bytes * sizeof(float) /1024/1024/1024;
        float gbps = total_bytes/eventMs * 1000 * 10;
        printf("sgemv_64x1_t1x1 [m=%d]==> %lf G Bytes/s, ms: %f\n", mm, gbps, eventMs);
   }


   for(int mm=1024; mm <=M; mm+=256){
        hipLaunchKernelGGL(sgemv_direct_64x1_t8x1, 
                        dim3(mm/64 ),
                        dim3(64),
                        0, 0,
                        deviceA ,deviceB ,deviceC, M, N, N);

        hipEventRecord(start, NULL);
        for (int i = 0; i < 10; i++){
          hipLaunchKernelGGL(sgemv_direct_64x1_t8x1, 
                        dim3(mm/64),
                        dim3(64),
                        0, 0,
                        deviceA ,deviceB ,deviceC, M, N, NN);

        }

        hipEventRecord(stop, NULL);
        hipEventSynchronize(stop);

        hipEventElapsedTime(&eventMs, start, stop);
  
        //printf("elapsed time:%f\n", eventMs);
        float total_bytes = ( double)(mm)* (double)N + double(N) + double(mm);          
        total_bytes = total_bytes * sizeof(float) /1024/1024/1024;
        float gbps = total_bytes/eventMs * 1000 * 10;
        printf("sgemv_64x1_t8x1 [m=%d]==> %lf G Bytes/s, ms: %f\n", mm, gbps, eventMs);
   }


  HIP_ASSERT(hipMemcpy(hostC, deviceC, M*sizeof(float), hipMemcpyDeviceToHost));

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
