Contents

[1      前言... 2](#_Toc15369680)

[1.1 HIP基础... 2](#_Toc15369681)

[1.2        预习资料... 3](#_Toc15369682)

[1.3        匹配硬件... 3](#_Toc15369683)

[2       基于Vega10的硬件相关优化实例... 3](#_Toc15369684)

[2.1  块与线程： Blocks & Threads. 3](#_Toc15369685)

[2.1.1 最高线程速率... 3](#_Toc15369686)

[2.1.2 1D形状 Block的线程速率曲线... 6](#_Toc15369687)

[2.1.3 2D 形状Block线程速率... 8](#_Toc15369688)

[2.1.3 3D 形状Block的线程生成速率... 9](#_Toc15369689)

[2.2 Compute Resources 计算资源... 10](#_Toc15369690)

[2.2.1         Execute 1,000,000 of FMA: 简单循环100万次... 10](#_Toc15369691)

[2.2.2 Specified Loop Unroll: 指定循环展开大小... 13](#_Toc15369692)

[2.2.3 Double Loop :双层循环... 13](#_Toc15369693)

[2.2.4  Increasing Threads In Parallel ：增加并行线程... 14](#_Toc15369694)

[2.2.5 Enough Parallel Threads: 足够多线程充满64个计算单元... 14](#_Toc15369695)

[2.3 VGPR: 矢量通用寄存器... 16](#_Toc15369696)

[2.4 SGPR： 标量通用寄存器... 17](#_Toc15369697)

[2.5 Divergence:  Wave 分歧... 20](#_Toc15369698)

[2.6 Memory Read Latency：显存读写延迟... 22](#_Toc15369699)

[2.6.1 L2 Cache Miss: 直接从显存读写... 22](#_Toc15369700)

[2.6.2 CacheLine Length: 缓存行长度... 23](#_Toc15369701)

[2.6.3 L1/L2 Cacheline Hit Latency：一/二级缓存命中延时... 24](#_Toc15369702)

[2.7 Alternative Method to measure CacheLine Size：另一组测试Cacheline长度... 25](#_Toc15369703)

[2.7.1测试CacheLine大小... 25](#_Toc15369704)

[2.7.2 Divergence for Memory Read/Write：显存访问分歧... 25](#_Toc15369705)

[2.8   NCHW-4D Index Generation: 4D数组索引生成... 25](#_Toc15369706)

[2.9 Local Data Share：本地数据共享... 26](#_Toc15369707)

[2.9.1 LDS Latency. 26](#_Toc15369708)

[2.9.2 LDS bank Conflicts. 27](#_Toc15369709)

[2.10 Memory Channel Conflicts：存储通道冲突... 28](#_Toc15369710)

[2.11 Math Functions：数学函数... 29](#_Toc15369711)

[2.12 Reduction：归约... 29](#_Toc15369712)

[2.13  Padding Before Convolution. 31](#_Toc15369713)

[2.13.1 1st Padding Kernel 31](#_Toc15369714)

[2.13.2 Optimize Kernel to Remove Scratch Memory. 33](#_Toc15369715)

[2.14 BatchNorm.. 33](#_Toc15369716)

[3      其他... 34](#_Toc15369717)

1     前言
========

1.1 HIP基础
---------

请参考HIP官方发布。 [https://github.com/ROCm-Developer-Tools/HIP](https://github.com/ROCm-Developer-Tools/HIP)

HIP允许并行程序开发者无缝移植CUDA C++代码。HIP源代码（包括从CUDA移植的HIP代码）可以被CUDA编译执行在 NVIDIA   GPU或者被HIPCC编译执行在AMD GPU上。HIP包括以下关键 特性：

*   HIP是一个轻量级的，它几乎不会对CUDA（或 hcc “HC”）代码造成性能影响，
*   HIP允许使用C++程序设计语言版本的多种特性编程，例如模板，C++11 Lambdas表达式，类，名字空间等。
*   HIP允许开发者使用基于目标平台的最佳开发环境和工具链。
*   “hipify”工具能够自动将CUDA源代码移植到HIP.
*   开发者可以指定平台（CUDA或 hcc）进行性能调试或者处理棘手问题。

1.2 预习资料
--------

在阅读第二章前，请确定已完成对以下材料的学习。

*   [HIP Kernel Language](https://github.com/ROCm-Developer-Tools/HIP/blob/master/docs/markdown/hip_kernel_language.md)
*   [HIP Runtime API (Doxygen)](http://rocm-developer-tools.github.io/HIP)
*   [HIP Porting Guide](https://github.com/ROCm-Developer-Tools/HIP/blob/master/docs/markdown/hip_porting_guide.md)
*   [HIP Porting Driver Guide](https://github.com/ROCm-Developer-Tools/HIP/blob/master/docs/markdown/hip_porting_driver_api.md)
*   [HIP Programming Guide](https://github.com/ROCm-Developer-Tools/HIP/blob/master/docs/markdown/hip_programming_guide.md)
*   Samples: [https://github.com/ROCm-Developer-Tools/HIP/tree/master/samples](https://github.com/ROCm-Developer-Tools/HIP/tree/master/samples)
*   Examples: [https://github.com/ROCm-Developer-Tools/HIP-Examples](https://github.com/ROCm-Developer-Tools/HIP-Examples)

1.3 匹配硬件
--------

本讲座中所有测试均基于AMD Radeon MI25或者硬件。如果改为其他硬件，需要修改计算核心的频率，Mi25对应的核心频率为1.536 Ghz。

2       基于Vega10的硬件相关优化实例
=========================

2.1  块与线程： Blocks & Threads
---------------------------

### 2.1.1 最高线程速率

AMD GCN硬件约定64 Threads 一个 wave，一个block可以有1-16个wave。硬件生成Threads的速率将直接影响最终程序的效率， 例如GPU显存的读写速度。 为了测试Vega10的Threads 速率， 我们可以写一个最简单的设备空函数,
```
__global__ void

null_kernel(hipLaunchParm lp,

       float* __restrict__ a)

{

}
```
执行rocm-smi，获得MI25的额定频率设置为1.536GHz。
<table><tr><td bgcolor=“#707070”>

========================        ROCm System Management Interface        ========================

================================================================================================

GPU   Temp   AvgPwr   SCLK    MCLK    PCLK           Fan     Perf    PwrCap   SCLK OD   MCLK OD  GPU%

0     69.0c  19.0W    1536Mhz 945Mhz  8.0GT/s, x16   12.94%  manual  220.0W   0%        0%       0%

================================================================================================

========================               End of ROCm SMI Log              ========================
</td></tr></table>
因此程序设置总的Threads 数量为 1024*1204*1024, 已获得接近秒级的GPU执行时间。

Threads速率是否与Block速率相关？这仍然是一个谜。因此测试程序暂时将每个 Block的Threads设置为最大值 1024。

为了获得准备的时间， 使用hipEventCreate函数产生两个事件 start, stop,通过hipEventRecord记录两个事件，并调用hipEventSynchronize确保stop是同步事件并被正确执行，hipEventElapsedTime(&eventMs, start, stop)函数将获得start, stop两个event的时间长度， 单位是毫秒。代码如下：
```
  hipEvent_t start, stop;

  hipEventCreate(&start);

  hipEventCreate(&stop);

  hipEventRecord(start, NULL);

  hipLaunchKernel(null_kernel,

                               dim3(1024*1024, 1),

                               dim3(1024, 1, 1), 

                              0, 0,

                               deviceA);

  hipEventRecord(stop, NULL);

  hipEventSynchronize(stop);

  hipEventElapsedTime(&eventMs, start, stop);
```
完整的代码如下：
```
//example-1a.cpp

#include <assert.h>

#include <stdio.h>

#include <algorithm>

#include <stdlib.h>

#include<iostream>

#include "hip/hip_runtime.h"

#define HIP_ASSERT(x) (assert((x)==hipSuccess))

#define TOTAL_THREADS  (1024*1024*1024)

#define NUM  1

#define THREADS_PER_BLOCK_X  1024

#define THREADS_PER_BLOCK_Y  1

#define THREADS_PER_BLOCK_Z  1

__global__ void

null_kernel(hipLaunchParm lp,

       float* __restrict__ a)

{

}

using namespace std;

int main() {

  float* hostA;

  float* deviceA;

  hipDeviceProp_t devProp;

  hipGetDeviceProperties(&devProp, 0);

  cout << " System minor " << devProp.minor << endl;

  cout << " System major " << devProp.major << endl;

  cout << " agent prop name " << devProp.name << endl;

  cout << "hip Device prop succeeded " << endl ;

  hipEvent_t start, stop;

  hipEventCreate(&start);

  hipEventCreate(&stop);

  float eventMs = 1.0f;

  int i;

  int errors;

  hostA = (float*)malloc(NUM * sizeof(float));

  HIP_ASSERT(hipMalloc((void**)&deviceA, NUM * sizeof(float)));

  hipLaunchKernel(null_kernel,

                  dim3(1, 1),

                  dim3(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y, THREADS_PER_BLOCK_Z),

                     0, 0,

                  deviceA);

  hipEventRecord(start, NULL);

  hipLaunchKernel(null_kernel,

                               dim3(TOTAL_THREADS/THREADS_PER_BLOCK_X, 1),

                               dim3(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y, THREADS_PER_BLOCK_Z),     

                     0, 0,

                               deviceA);

  hipEventRecord(stop, NULL);

  hipEventSynchronize(stop);

  hipEventElapsedTime(&eventMs, start, stop);

  printf("kernel_time (hipEventElapsedTime) =%6.3fms\\n", eventMs);

  printf("Threads_per_cycle for Vega10 - 1.536GHz = % 3d\\n", int(TOTAL_THREADS / eventMs / 1.536 / 1e6));

  HIP_ASSERT(hipFree(deviceA));

  free(hostA);

  return errors;

}
```
使用如下指令编译  example-1a.cpp

*  <table><tr><td bgcolor=“#707070”> hipcc example-1a.cpp -o example-1a.exe </td></tr></table>

本人假定随后章节采用相同的方法进行编译。

执行example-1a.exe，得到如下结果：
<table><tr><td bgcolor=“#707070”>
 System minor 0

 System major 3

 agent prop name Device 687f

hip Device prop succeeded

kernel_time (hipEventElapsedTime) =10.890ms

Threads_per_cycle for Vega10 - 1.536GHz =  64
</td></tr></table>
结果说明Mi25获得64 threads/Cycle的极限性能。

### 2.1.2 1D形状 Block的线程速率曲线

     第一个简单测试获得MI25的线程速率为 64 threads/cycle,那么是不是所有1D 形状块均可获得极限速率呢？

      Example2.cpp 将测试 自小而大不同的BlockDim, Dim3(1,1,1),  Dim3(2,1,1), Dim3(4,1,1),Dim3(8,1,1), …,(1024,1,1)。获得如下结果:
<table><tr><td bgcolor=“#707070”>
 System minor 0

 System major 3

 agent prop name Device 687f

hip Device prop succeeded

kernel_time (hipEventElapsedTime) =2789.162ms

threads_per_block = 1,Threads_per_cycle for Vega10 - 1.536GHz =   0

kernel_time (hipEventElapsedTime) =1395.156ms

threads_per_block = 2,Threads_per_cycle for Vega10 - 1.536GHz =   1

kernel_time (hipEventElapsedTime) =697.689ms

threads_per_block = 4,Threads_per_cycle for Vega10 - 1.536GHz =   1

kernel_time (hipEventElapsedTime) =348.875ms

threads_per_block = 8,Threads_per_cycle for Vega10 - 1.536GHz =   2

kernel_time (hipEventElapsedTime) =174.456ms

threads_per_block = 16,Threads_per_cycle for Vega10 - 1.536GHz =   4

kernel_time (hipEventElapsedTime) =87.238ms

threads_per_block = 32,Threads_per_cycle for Vega10 - 1.536GHz =   8

kernel_time (hipEventElapsedTime) =43.629ms

threads_per_block = 64,Threads_per_cycle for Vega10 - 1.536GHz =  16

kernel_time (hipEventElapsedTime) =21.828ms

threads_per_block = 128,Threads_per_cycle for Vega10 - 1.536GHz =  32

kernel_time (hipEventElapsedTime) =10.929ms

threads_per_block = 256,Threads_per_cycle for Vega10 - 1.536GHz =  64

kernel_time (hipEventElapsedTime) =10.914ms

threads_per_block = 512,Threads_per_cycle for Vega10 - 1.536GHz =  64

kernel_time (hipEventElapsedTime) =10.909ms

threads_per_block = 1024,Threads_per_cycle for Vega10 - 1.536GHz =  64
</td></tr></table>
仔细观察，仅仅当 BlockDim = 256， 512, 1024时， 线程产生速度达到峰值。这个信息有什么含义， 或者对GPU程序优化有何指导意义？

举例， 在深度学习中有大量的简单操作， 例如Copy,  激活函数，如果程序使用了比256小的BlockDim, 那么程序将很难达到理论值,   例如64，那么理论极限很有可能是64/256。深度学习经常使用Padding Copy, 如果 H x W = 7x7,  Padding= 3, 那么理论极限将会是13*13/256 = 66%。

以上两种情况， 如果程序能够将原来4 threads的工作合并到一个thread，每个线程处理的事务随之提高到4倍，例如读写操作，将极大地提高理论极限。
<table><tr><td bgcolor=“#707070”>
Case1 :  min ( 64 *4, 256 )        = 256

Case 2:  min ( 13 * 13 *4, 256) = 256
</td></tr></table>
这个测试结果是否有值得怀疑的地方？ 这个测试结果证明只有BlockDim =256才能达到理论极限，和AMD GCN的图形像素渲染能力不匹配，颜色渲染能力达到了64 Pixels/Cycle。GCN架构的Pixel Shader都是64个 像素一个Wave，换而言之HIP 也应该能够达到64 Threads/Cycle。而测试结果只有Pixel Shader的1/4，这有两种可能： 1） ROCm使用了特别的寄存器设置使得线程产生速度降低到了1/4；2）硬件的计算线程生成速度是像素着色器的1/4速度。第二个原因的可能性比较小，GCN统一化的着色器架构设计应保证不同类型的着色器（几何，像素，计算）线程速度相同， 否则对应硬件资源将被浪费。

### 2.1.3 2D 形状Block线程速率

本节将测试2D 形状Block 的线程速率，前两节已知1D最大线程数为1024，那么对应最大的 BlockDim应该为 Dim3(32, 32,1),  最小为Dim3(1,1,1)，这样可以组成32个不同的测试组合。

编译执行eaxaple-1c.cpp，得到如下结果。
<table><tr><td bgcolor=“#707070”>
threads_per_block = [1,1,1],Threads_per_cycle for Vega10 - 1.536GHz =   0

threads_per_block = [2,2,1],Threads_per_cycle for Vega10 - 1.536GHz =   1

threads_per_block = [3,3,1],Threads_per_cycle for Vega10 - 1.536GHz =   2

threads_per_block = [4,4,1],Threads_per_cycle for Vega10 - 1.536GHz =   4

threads_per_block = [5,5,1],Threads_per_cycle for Vega10 - 1.536GHz =   6

threads_per_block = [6,6,1],Threads_per_cycle for Vega10 - 1.536GHz =   9

threads_per_block = [7,7,1],Threads_per_cycle for Vega10 - 1.536GHz =  12

threads_per_block = [8,8,1],Threads_per_cycle for Vega10 - 1.536GHz =  16

threads_per_block = [9,9,1],Threads_per_cycle for Vega10 - 1.536GHz =  20

threads_per_block = [10,10,1],Threads_per_cycle for Vega10 - 1.536GHz =  25

threads_per_block = [11,11,1],Threads_per_cycle for Vega10 - 1.536GHz =  30

threads_per_block = [12,12,1],Threads_per_cycle for Vega10 - 1.536GHz =  36

threads_per_block = [13,13,1],Threads_per_cycle for Vega10 - 1.536GHz =  42

threads_per_block = [14,14,1],Threads_per_cycle for Vega10 - 1.536GHz =  49

threads_per_block = [15,15,1],Threads_per_cycle for Vega10 - 1.536GHz =  56

threads_per_block = [16,16,1],Threads_per_cycle for Vega10 - 1.536GHz =  64

threads_per_block = [17,17,1],Threads_per_cycle for Vega10 - 1.536GHz =  58

threads_per_block = [18,18,1],Threads_per_cycle for Vega10 - 1.536GHz =  54

threads_per_block = [19,19,1],Threads_per_cycle for Vega10 - 1.536GHz =  60

threads_per_block = [20,20,1],Threads_per_cycle for Vega10 - 1.536GHz =  57

threads_per_block = [21,21,1],Threads_per_cycle for Vega10 - 1.536GHz =  63

threads_per_block = [22,22,1],Threads_per_cycle for Vega10 - 1.536GHz =  60

threads_per_block = [23,23,1],Threads_per_cycle for Vega10 - 1.536GHz =  59

threads_per_block = [24,24,1],Threads_per_cycle for Vega10 - 1.536GHz =  64

threads_per_block = [25,25,1],Threads_per_cycle for Vega10 - 1.536GHz =  62

threads_per_block = [26,26,1],Threads_per_cycle for Vega10 - 1.536GHz =  61

threads_per_block = [27,27,1],Threads_per_cycle for Vega10 - 1.536GHz =  61

threads_per_block = [28,28,1],Threads_per_cycle for Vega10 - 1.536GHz =  60

threads_per_block = [29,29,1],Threads_per_cycle for Vega10 - 1.536GHz =  60

threads_per_block = [30,30,1],Threads_per_cycle for Vega10 - 1.536GHz =  60

threads_per_block = [31,31,1],Threads_per_cycle for Vega10 - 1.536GHz =  60

threads_per_block = [32,32,1],Threads_per_cycle for Vega10 - 1.536GHz =  64
</td></tr></table>
结果清晰第显示，只有当BlockDim的总线程数量是256的倍数，Dim3(16,16,1), Dim3(24,24,1), Dim3(32,32,1)，才能获得极限线程生成速率。Dim3(32,16,1)读者有兴趣可以自己测试。

对于HIP程序开发者，对于简单的显存读写类，建议使用256倍数的BlockDim以获取最高线程生成速率。计算异常密集的任务，它的性能主要瓶颈和线程生成速率无关时，建议使用64倍数的BlockDim。

### 2.1.3 3D 形状Block的线程生成速率

HIP也提供3D 形状的Block,  1024最大线程数转化为三维形状，可以为Dim( 16,16,4), Dim( 32,16,2),  Dim(8,8,64)等。下面我们选择一些特殊形状， 测试其性能变化，Dim3(1,1,1)，Dim3(2,2,2), Dim3(3,3,3)，Dim3(4,4,4)，Dim3(5,5,5)，Dim3(6,6,6), Dim3(7,7,7)，Dim3(8,8,8)，Dim3(9,9,9)和Dim3(10,10,10)。

编译执行example-1d.cpp。得到如下结果。
<table><tr><td bgcolor=“#707070”>
threads_per_block = [1,1,1],Threads_per_cycle for Vega10 - 1.536GHz =   0

threads_per_block = [2,2,2],Threads_per_cycle for Vega10 - 1.536GHz =   2

threads_per_block = [3,3,3],Threads_per_cycle for Vega10 - 1.536GHz =   7

threads_per_block = [4,4,4],Threads_per_cycle for Vega10 - 1.536GHz =  16

threads_per_block = [5,5,5],Threads_per_cycle for Vega10 - 1.536GHz =  31

threads_per_block = [6,6,6],Threads_per_cycle for Vega10 - 1.536GHz =  54

threads_per_block = [7,7,7],Threads_per_cycle for Vega10 - 1.536GHz =  57

threads_per_block = [8,8,8],Threads_per_cycle for Vega10 - 1.536GHz =  64

threads_per_block = [9,9,9],Threads_per_cycle for Vega10 - 1.536GHz =  61

threads_per_block = [10,10,10],Threads_per_cycle for Vega10 - 1.536GHz =  62
</td></tr></table>
这个实例的结论和前两个测试相同， 只用线程数为256的整倍数才能获得最佳性能。

2.2 Compute Resources 计算资源
--------------------------

Vega64有64个计算单元（compute unit），每个计算单元有64个乘加器。那么每个计算单元能够64 FMAs/Cycle，64个计算单元的能力为4096 cycles/ cycle，每个乘法包含一个乘法和加法，算做两个浮点运算，乘以频率1.536Ghz =  15.6T Flops/s。我们下面将研究HIPCC如何在单个计算单元获得64 FMAs /cycle.

### 2.2.1        Execute 1,000,000 of FMA: 简单循环100万次

 256 threads执行100万次FMA，只有64个乘加器，那么每个乘加器需要执行400万条指令，那么执行时间最短时间为 4/1.536 = 2.6毫秒。编译器通常带有许多有优化技术，它会优化掉对最终结果无贡献的大量计算，因此程序必须迷惑编译器，假装程序一定会产生输出。
```
#define FMA_PER_THREADS       1000000

__global__ void

test_kernel(hipLaunchParm lp,

       float* __restrict__ a)

{

       int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;

       int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;      

       float t0 = (float)x / (float) (x + 1);

       float t1 = float(y + 1) / (float)(y + 100000000);

       float sum=0.0;

       for(int i =0; i < FMA_PER_THREADS;i++)

       {

              sum = t0 *sum + t1;

       }

       //迷惑编译器，防止编译器优化将上面一百万条指令全部移除

       if( (float(x)+sum) < -1.0f)

       {

              a[0] = sum;

       }

}
```
完整的程序参考example-2a.cpp。使用如下命令行编译：

<table><tr><td bgcolor=“#707070”>
hipcc example-2a.cpp -o example-2a.exe 
</td></tr></table>

hcc 提供了一个反汇编工具 /opt/rocm/hcc/bin/extractkernel。用如下命令获得上述test_kernel的GCN汇编代码：

<table><tr><td bgcolor=“#707070”>
extractkernel -i  ./example-2a.exe
</td></tr></table>
执行命令得到的输出：
<table><tr><td bgcolor=“#707070”>
Generated GCN ISA for gfx900 at: ./example-2a.exe-gfx900.isa
</td></tr></table>
打开example-2a.exe-gfx900.isa，可以发现如下代码段：
<table><tr><td bgcolor=“#707070”>
000000000000124c BB0_1:

       v_mad_f32 v3, v1, v3, v2                                   // 00000000124C: D1C10003 040A0701

       v_mad_f32 v3, v1, v3, v2                                   // 000000001254: D1C10003 040A0701

       v_mad_f32 v3, v1, v3, v2                                   // 00000000125C: D1C10003 040A0701

       v_mad_f32 v3, v1, v3, v2                                   // 000000001264: D1C10003 040A0701

       v_mad_f32 v3, v1, v3, v2                                   // 00000000126C: D1C10003 040A0701

       v_mad_f32 v3, v1, v3, v2                                   // 000000001274: D1C10003 040A0701

       v_mad_f32 v3, v1, v3, v2                                   // 00000000127C: D1C10003 040A0701

       v_mad_f32 v3, v1, v3, v2                                   // 000000001284: D1C10003 040A0701

       v_mad_f32 v3, v1, v3, v2                                   // 00000000128C: D1C10003 040A0701

       v_mad_f32 v3, v1, v3, v2                                   // 000000001294: D1C10003 040A0701

       v_mad_f32 v3, v1, v3, v2                                   // 00000000129C: D1C10003 040A0701

       v_mad_f32 v3, v1, v3, v2                                   // 0000000012A4: D1C10003 040A0701

       v_mad_f32 v3, v1, v3, v2                                   // 0000000012AC: D1C10003 040A0701

       v_mad_f32 v3, v1, v3, v2                                   // 0000000012B4: D1C10003 040A0701

       v_mad_f32 v3, v1, v3, v2                                   // 0000000012BC: D1C10003 040A0701

       v_mad_f32 v3, v1, v3, v2                                   // 0000000012C4: D1C10003 040A0701

       v_mad_f32 v3, v1, v3, v2                                   // 0000000012CC: D1C10003 040A0701

       v_mad_f32 v3, v1, v3, v2                                   // 0000000012D4: D1C10003 040A0701

       v_mad_f32 v3, v1, v3, v2                                   // 0000000012DC: D1C10003 040A0701

       v_mad_f32 v3, v1, v3, v2                                   // 0000000012E4: D1C10003 040A0701

       v_mad_f32 v3, v1, v3, v2                                   // 0000000012EC: D1C10003 040A0701

       v_mad_f32 v3, v1, v3, v2                                   // 0000000012F4: D1C10003 040A0701

       v_mad_f32 v3, v1, v3, v2                                   // 0000000012FC: D1C10003 040A0701

       v_mad_f32 v3, v1, v3, v2                                   // 000000001304: D1C10003 040A0701

       v_mad_f32 v3, v1, v3, v2                                   // 00000000130C: D1C10003 040A0701

       v_mad_f32 v3, v1, v3, v2                                   // 000000001314: D1C10003 040A0701

       v_mad_f32 v3, v1, v3, v2                                   // 00000000131C: D1C10003 040A0701

       v_mad_f32 v3, v1, v3, v2                                   // 000000001324: D1C10003 040A0701

       v_mad_f32 v3, v1, v3, v2                                   // 00000000132C: D1C10003 040A0701

       v_mad_f32 v3, v1, v3, v2                                   // 000000001334: D1C10003 040A0701

       v_mad_f32 v3, v1, v3, v2                                   // 00000000133C: D1C10003 040A0701

       v_mad_f32 v3, v1, v3, v2                                   // 000000001344: D1C10003 040A0701

       v_mad_f32 v3, v1, v3, v2                                   // 00000000134C: D1C10003 040A0701

       v_mad_f32 v3, v1, v3, v2                                   // 000000001354: D1C10003 040A0701

       v_mad_f32 v3, v1, v3, v2                                   // 00000000135C: D1C10003 040A0701

       v_mad_f32 v3, v1, v3, v2                                   // 000000001364: D1C10003 040A0701

       v_mad_f32 v3, v1, v3, v2                                   // 00000000136C: D1C10003 040A0701

       v_mad_f32 v3, v1, v3, v2                                   // 000000001374: D1C10003 040A0701

       v_mad_f32 v3, v1, v3, v2                                   // 00000000137C: D1C10003 040A0701

       s_sub_i32 s2, s2, 40                                       // 000000001384: 8182A802

       s_cmp_lg_u32 s2, 0                                         // 000000001388: BF078002

       v_mad_f32 v3, v1, v3, v2                                   // 00000000138C: D1C10003 040A0701

       s_cbranch_scc1 BB0_1                                       // 000000001394: BF85FFAD
</td></tr></table>
该段GCN 汇编代码是对应test_kernel的100万次循环，包含：

*   40个v_mad_f32指令，编译器做了默认40次循环展开，
*   两条SALU, s_sub_i32, s_cmp_lg_u32
*   一条跳转指令 s_cbranch_scc1

那么对应FMA指令的有效率为， 40/43 = 93%，乘以每个计算单元的64个乘加器，理论上可以获得59个FMA /Cycle.

现在执行example-2a.exe获得测试性能。
<table><tr><td bgcolor=LightGray>
Total Threads = 1 * 256, FMA_per_cycle for Vega10 - 1.536GHz =     44
</td></tr></table>
实际上测试程序使用256 threads仅仅获得了44个FMA/Cycle，远远低于理论预期。那么这里存在一些我们还没发现的性能陷阱。可以有两个方向进行测试，例如采用两层循环，控制循环展开的指令数目， 增加threads数目以提高并行性，并减少因指令缓存(instruction Cache)读取失败的机率。

### 2.2.2 Specified Loop Unroll: 指定循环展开大小

指定循环展开块的大小可以减少SVALU的比例，提高程序整体效率，我们来尝试指定循环展开数量为100。代码如下：
```
#pragma unroll 100

       for(int i =0; i < FMA_PER_THREADS;i++)

       {

              sum = t0 *sum + t1;

       }
```

编译example-2b.cpp并执行获得如下结果。
<table><tr><td bgcolor=“#707070”>
Total Threads = 1 * 256, FMA_per_cycle for Vega10 - 1.536GHz =     48
</td></tr></table>
成绩从44 FMA/Cycle/CU 提高到了48 FMA/Cycle/CU。继续使用extractkernels来检查GCN汇编代码，我们发现主体循环代码包含：

*   100个v_mad_f32指令，完全匹配指定的循环展开次数100次
*   两条SALU, s_addk_i32, s_cmp_lg_u32
*   一条跳转指令

此时example-2b能获得理论性能为100/103 * 64 = 62 FMA/cycle/CU， example-2a高3 FMA/Cycle/CU，实际获得4 FMA/Cycle/CU的提升。实际效果良好。但是距离我们期待的 64 FMA/Cycle/CU仍然有比较大的差距。

### 2.2.3 Double Loop :双层循环

  Example-2c将尝试多层循环，内存循环体使用100次循环，外层循环体10000次循环。
```
       for(int i =0; i < FMA_PER_THREADS/100;i++)

       {

              for(int j=0; j < 100; j++)

              sum = t0 *sum + t1;

       }
```
编译执行example-2c.cpp得到如下输出结果：
<table><tr><td bgcolor=“#707070”>
Total Threads = 1 * 256, FMA_per_cycle for Vega10 - 1.536GHz =     59 
</td></tr></table>
性能得到了很大提升，以惯例继续使用extractkernel查看主要循环体：

*   100个v_mad_f32指令，完全匹配内层循环体100次
*   两条SALU, s_add_i32, s_cmp_lg_u32
    *   s_add_i32 s2, s2, -1
*   一条跳转指令s_cbranch_scc1

      这个结果很难解释为何example-3c.cpp 比example-3b.cpp获得大幅度的性能提升。仔细检查example-2b和example-2c的GCN汇编代码，另外一个微小区别是整个Kernel代码段的长度差了4个字节。一个可能测猜测是Instruction Cache有特定的尺寸，对于性能影响很大，如果整个循环体代码长度是Instruction Cache的完整倍数，那么将获得最优性能，否则最终的性能为实际指令编码的字节数与对应Cacheline之比。例如Instruction Cache为8 个DWORD，那么整个循环体最多损失14 DWORDs，103条指令编码总共203个DWORDs, 最少26条Cachelines，最多27条Cachelines，如果多一个不对齐的Cahceline, 那么最多损失8%的性能，或者5-6条FMA/Cycle/CU。如果Instruction Cache Line有两条不对齐的Cachelines，最大性能差距会达到11条 FMA/Cycle/CU。

### 2.2.4  Increasing Threads In Parallel ：增加并行线程

256 threads意味着每个乘加器只有一个线程， 如果将每个乘加器的线程数量增加到2个，这样每个乘加器可以乒乓线程以隐藏延迟，是否能够提高计算单元的效率？

编译并执行Example-2d.cpp，获得如下结果。
<table><tr><td bgcolor=“#707070”>
Total Threads = 1 * 256, FMA_per_cycle for Vega10 - 1.536GHz =     59

Total Threads = 1 * 512, FMA_per_cycle for Vega10 - 1.536GHz =     62

Total Threads = 1 * 768, FMA_per_cycle for Vega10 - 1.536GHz =     63

Total Threads = 1 * 1024, FMA_per_cycle for Vega10 - 1.536GHz =     63
</td></tr></table>
 结果显示，当我们增加1个计算单元的并行线程数，能够有效增加SIMD的效率。

### 2.2.5 Enough Parallel Threads: 足够多线程充满64个计算单元

前面四节讨论了如何获取单个计算单元的峰值性能，如果想要达到最佳性能，一个可能的办法是手写GCN assembly，然后仔细调整循环体Cacheline的长度，使得Assembly Shader无限接近理论最高性能。

这节我们将探究不同 Block数量对于性能的影响。下面这段程序使用双重循环测试峰值计算性能，Block从1，2，3, …, 128，BlockDim可选取 Dim3(256,1,1), Dim3(512, 1,1), Dim3(768,1,1)和 Dim3(1024,1,1)。
```
  for (int i = 1; i < 5; i = i + 1) {

     for (int j = 0; j < 129; j++)

     {

          hipEventRecord(start, NULL);

          hipLaunchKernel(null_kernel,

                                 dim3(j, 1, 1),

                                 dim3(THREADS_PER_BLOCK_X * i, 1, 1),

                                 0, 0,

                                 deviceA);

          hipEventRecord(stop, NULL);

          hipEventSynchronize(stop);

          hipEventElapsedTime(&eventMs, start, stop);

          printf("kernel_time (hipEventElapsedTime) =%6.3fms\\n", eventMs);

          double FMA_per_cycle = double(THREADS_PER_BLOCK_X) * i *j * double(FMA_PER_THREDS) / eventMs / (1.536 * 1e6) + 0.5;

            printf("Total Threads = %d * %d, FMA_per_cycle for Vega10 - 1.536GHz = %6d\\n", j, THREADS_PER_BLOCK_X * i,    

                     (int)FMA_per_cycle);

        }

  }
```
编译执行example-2e.cpp将得到4x128=512不同的性能组合， 我们选取其中的10个组合。
<table><tr><td bgcolor=“#707070”>
kernel_time (hipEventElapsedTime) =10.630ms

Total Threads = 1 * 1024, FMA_per_cycle for Vega10 - 1.536GHz =     63

kernel_time (hipEventElapsedTime) =10.639ms

Total Threads = 2 * 1024, FMA_per_cycle for Vega10 - 1.536GHz =    125

kernel_time (hipEventElapsedTime) =10.641ms

Total Threads = 3 * 1024, FMA_per_cycle for Vega10 - 1.536GHz =    188

Total Threads = 8 * 1024, FMA_per_cycle for Vega10 - 1.536GHz =    499

kernel_time (hipEventElapsedTime) =10.720ms

Total Threads = 16 * 1024, FMA_per_cycle for Vega10 - 1.536GHz =    995

kernel_time (hipEventElapsedTime) =10.803ms

Total Threads = 32 * 1024, FMA_per_cycle for Vega10 - 1.536GHz =   1975

kernel_time (hipEventElapsedTime) =10.963ms

Total Threads = 64 * 1024, FMA_per_cycle for Vega10 - 1.536GHz =   3892

kernel_time (hipEventElapsedTime) =21.376ms

Total Threads = 65 * 1024, FMA_per_cycle for Vega10 - 1.536GHz =   2027

kernel_time (hipEventElapsedTime) =21.383ms

Total Threads = 66 * 1024, FMA_per_cycle for Vega10 - 1.536GHz =   2058

kernel_time (hipEventElapsedTime) =21.386ms
</td></tr></table>
我们观察到Block数量从1到64，程序执行时间几乎不变，GPU的FMA速率线性增长， 而Block数量增加到65，GPU执行时间增加一倍，表明Vega10 GPU总共有64个计算单元。我们在做程序优化的时候，程序需要尽可能保证Block的总数量是64的整倍数，这样能够保证减少因为计算单元空闲造成的性能下降， 例如总共65个Block，那么它的最大理论效率只有64/128 = 50.8%。性能基准测试程序期望压榨每一个百分点的性能，Block总数将会成为成为性能优化的一个不可忽视手段。

2.3 VGPR: 矢量通用寄存器
-----------------

  上节我们讨论了计算单元和并行线程数的关系，并且分析了Instruction Cacheline对于性能的影响。每个计算线程还有非常重要的资源—VPGRs。当Kernel使用的VGPR资源过多， 就会造成只有一个Thread运行在对应的MAC，或者单一wave（64 threads）运行在一个SIMD，那么会造成严重的性能下降。如果线程使用的VGPR超过了硬件最大资源，编译器将会开辟一块内存，将超出部分暂时缓存到GPU显存，性能可能会下降到峰值性能的5%以下。

测试最大VGPR有很多方法， 例如构造一个VPGR的二叉树，防止编译器优化减少VGPR的数量，每次增加二叉树叶子节点的数量，指导性能剧烈突然下降为止。我这里采用另外一个简单方法，rocm 提供了一个内嵌汇编的方式，下面的这个 Kernel测试最大VGPR是否为V255，如果能够编译成功，那么可以VGPR总数为256。然后逐渐增大VGPR索引，看看是否编译无法通过，或者执行失败，那么上一个成功的索引值就是最大VGPR。

下面是一个测试VGPR的简单实例。
```
__global__ void

test_kernel_255(hipLaunchParm lp,

       float* __restrict__ a)

{

       asm volatile("v_mov_b32 v0, 0");

       asm volatile("v_mov_b32 v255, 0" );

}
```
 我们尝试编译并执行example-3a.cpp。编译和执行都顺利完成。然后再次用神器extractkernel查看 GCN assembly shader。发现程序只有如下三行代码：
```
              v_mov_b32_e32 v0, 0                                        // 000000001100: 7E000280

              v_mov_b32_e32 v255, 0                                    // 000000001104: 7FFE0280

              s_endpgm                                                            // 000000001108: BF810000
```
这个结果非常符合我们的预期。我们可以增加下面一个Kernel到example-3b.cpp
```
__global__ void

test_kernel_256(hipLaunchParm lp,

       float* __restrict__ a)

{

       asm volatile("v_mov_b32 v0, 0");

       asm volatile("v_mov_b32 v256, 0");

}
```
老规矩，调用 hipcc尝试编译example-3b.cpp。编译失败并获得下面错误信息：
<table><tr><td bgcolor=“#707070”>
<inline asm>:1:16: error: unknown token in expression

        v_mov_b32 v256, 0

                      ^

note: !srcloc = 833

<inline asm>:1:18: error: not a valid operand.

        v_mov_b32 v256, 0

                        ^

note: !srcloc = 833

Generating AMD GCN kernel failed in llc for target: gfx900

clang-8: error: linker command failed with exit code 1 (use -v to see invocation)
</td></tr></table>
这个kernel有两个不同的内嵌汇编，第一条成功而第二条失败，表明Vega10能够支持的最大VGPR为256（从V0开始计数为V255）。

2.4 SGPR： 标量通用寄存器
-----------------

SGPR在AMD GCN体系结构是非常重要的一项特性。SGPR第一个用途是读GPU显存常量到计算单元，例如图形渲染中的投影矩阵，纹理对象描述，纹理采样描述等。SGPR是可读可写， 它可以作为用于程序流程控制，例如循环变量， 从而减低SIMD VGPR的需求，同时也降低大部分循环控制的功耗。

同VGPR一样，SGPR资源也是有限的， 我们也可以采用内联汇编的方法测试最大SGPR。VGPR越界在编译的时候直接出错，理论SGPR也有同样的性质。Example-4a.cpp使用下面的Kernel寻找最大SGPR。
<table><tr><td bgcolor=“#707070”>
__global__ void

test_kernel_255(hipLaunchParm lp,

       float* __restrict__ a)

{

   asm volatile("s_mov_b32 s0, 0");

   asm volatile("s_mov_b32 s95, 0" );

   asm volatile("s_mov_b32 s96, 0" );

   asm volatile("s_mov_b32 s97, 0" );

   asm volatile("s_mov_b32 s98, 0" );

   asm volatile("s_mov_b32 s99, 0" );

   asm volatile("s_mov_b32 s100, 0" );

   asm volatile("s_mov_b32 s101, 0" );

   asm volatile("s_mov_b32 s102, 0" );

   asm volatile("s_mov_b32 s103, 0" );

   asm volatile("s_mov_b32 s104, 0" );

   asm volatile("s_mov_b32 s105, 0" );

   asm volatile("s_mov_b32 s106, 0" );

   asm volatile("s_mov_b32 s107, 0" );

   asm volatile("s_mov_b32 s108, 0" );

   asm volatile("s_mov_b32 s109, 0" );

}
</td></tr></table>
老规矩，使用“hipcc  example-4a.cpp -o example-4a.exe”尝试编译。 得到如下错误：
<table><tr><td bgcolor=“#707070”>
<inline asm>:1:16: error: unknown token in expression

        s_mov_b32 s102, 0

                      ^

note: !srcloc = 950

<inline asm>:1:18: error: not a valid operand.

        s_mov_b32 s102, 0

                        ^

note: !srcloc = 950

<inline asm>:1:16: error: unknown token in expression

        s_mov_b32 s103, 0

                      ^

note: !srcloc = 990

<inline asm>:1:18: error: not a valid operand.

        s_mov_b32 s103, 0

                        ^

note: !srcloc = 990

<inline asm>:1:16: error: unknown token in expression

        s_mov_b32 s104, 0

                      ^

note: !srcloc = 1030

<inline asm>:1:18: error: not a valid operand.

        s_mov_b32 s104, 0

                        ^

note: !srcloc = 1030

<inline asm>:1:16: error: unknown token in expression

        s_mov_b32 s105, 0

                      ^

note: !srcloc = 1070

<inline asm>:1:18: error: not a valid operand.

        s_mov_b32 s105, 0

                        ^

note: !srcloc = 1070

<inline asm>:1:16: error: unknown token in expression

        s_mov_b32 s106, 0

                      ^

note: !srcloc = 1110

<inline asm>:1:18: error: not a valid operand.

        s_mov_b32 s106, 0

                        ^

note: !srcloc = 1110

<inline asm>:1:16: error: unknown token in expression

        s_mov_b32 s107, 0

                      ^

note: !srcloc = 1150

<inline asm>:1:18: error: not a valid operand.

        s_mov_b32 s107, 0

                        ^

note: !srcloc = 1150

<inline asm>:1:16: error: unknown token in expression

        s_mov_b32 s108, 0

                      ^

note: !srcloc = 1190

<inline asm>:1:18: error: not a valid operand.

        s_mov_b32 s108, 0

                        ^

note: !srcloc = 1190

<inline asm>:1:16: error: unknown token in expression

        s_mov_b32 s109, 0

                      ^

note: !srcloc = 1230

<inline asm>:1:18: error: not a valid operand.

        s_mov_b32 s109, 0

                        ^

note: !srcloc = 1230

Generating AMD GCN kernel failed in llc for target: gfx900

clang-8: error: linker command failed with exit code 1 (use -v to see invocation)
</td></tr></table>
SGPR S102之前能够被编译器正确识别，我们就找到了最大程序SGPR为 S101(从S0开始计数)。在GCN 体系结构设计中，SGPR资源始终可以用到SGPR 101 ，读者可以用BlockDim=Dim3(1024,1,1)进行验证，而VGPR在BlockDim=Dim3(1024,1,1)则下降到 V63。

2.5 Divergence:  Wave 分歧
------------------------

在SIMD结构中， 有一种特殊的情况， 如果一个wave只有1个Thread和其他63个Threads执行路径不同，那么对性能有何影响，例如我们把2.2.1的代码修改如下：
```
       if (hipThreadIdx_x == 0) {

              for (int i = 0; i < FMA_PER_THREDS; i++){

                      sum = t0 * sum + t1;

              }

       }

       else {

              for (int i = 0; i < FMA_PER_THREDS; i++){

                      sum = t1 * sum + t0;

              }

       }
```
SIMD的特点是所有Threads必须执行相同的指令， 由于Thread0和其他代码路径不同， 那么编译器必须先生成Thread0的代码，然后生成剩余63个Threads的代码。那么SIMD则顺序Thread0的代码，然后Thread1-63的代码。那么性能将下降到2.2.1实例代码的50%。

是否可以改进这种分歧？把2.2.1的实例中循环体部分看作一个函数 foo，那么Thread0可以当作foo（t0, t1），thread1-63看做是foo(t1,t0)，通过对参数的交换，实现所有线程调用同样参数，那么可以大大降低Divergence带来的性能下降。 参考下面test_kernel_optimize.
```
__global__ void

test_kernel_divergence(hipLaunchParm lp,

       float* __restrict__ a)

{

       int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;

       int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;

       float t0 = (float)x / (float)(x + 1);

       float t1 = float(y + 1) / (float)(y + 100000000);

       float sum = 0.0;

       if (hipThreadIdx_x == 0) {

              for (int i = 0; i < FMA_PER_THREDS; i++){

                      sum = t0 * sum + t1;

              }

       }

       else {

              for (int i = 0; i < FMA_PER_THREDS; i++){

                      sum = t1 * sum + t0;

              }

       }

       if ((float(x) + sum) < -1.0f)

       {

              a[0] = sum;

       }

}

__global__ void

test_kernel_optimize(hipLaunchParm lp,

       float* __restrict__ a)

{

       int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;

       int y = hipBlockDim_y * hipBlockIdx_y + hipThreadIdx_y;

       float t0 = (float)x / (float)(x + 1);

       float t1 = float(y + 1) / (float)(y + 100000000);

       float sum = 0.0;

       if (hipThreadIdx_x == 0) {

              float t = t0;

              t1 = t0;

              t0 = t;

       }

       for (int i = 0; i < FMA_PER_THREDS ; i++)

       {

              sum = t0 * sum + t1;

       }

       if ((float(x) + sum) < -1.0f)

       {

              a[0] = sum;

       }

}
```
编译并执行程序example-5a.cpp得到如下结果，上述理论得到了验证。
<table><tr><td bgcolor=“#707070”>
execute test kernel

kernel_time (hipEventElapsedTime) = 3.774ms

Total Threads = 1 * 256, FMA_per_cycle for Vega10 - 1.536GHz =     44

execute divergence kernel

kernel_time (hipEventElapsedTime) = 8.119ms

Total Threads = 1 * 256, FMA_per_cycle for Vega10 - 1.536GHz =     21

execute optimized kernel

kernel_time (hipEventElapsedTime) = 3.838ms

Total Threads = 1 * 256, FMA_per_cycle for Vega10 - 1.536GHz =     43
</td></tr></table>
2.6 Memory Read Latency：显存读写延迟
------------------------------

### 2.6.1 L2 Cache Miss: 直接从显存读写

读显存的延迟可以连续读不同的Cacheline，下一次读操作用前一次读操作的返回值，连续执行1,000,000次的有依赖关系的读操作，取平均即可获得读操作的延迟。我们目前还不知道如何Cacheline大小，而依据经验值，一条cacheline长度 可能为 16，32，64，128字节，因此我们程序读下一个值的地址比上一个地址大256DWORDs（1024字节），这样可以保证整个程序不会读两个相同的Cacheline。程序中buf的所有值为256。
```
__global__ void

test_kernel(hipLaunchParm lp,

       int* __restrict__ buf)

{

       int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;

int t = buf[x];

       //dependency reads

       for( int i=1; i < MAX_MEM_READS; i++)

       {

          t = buf[t * i ];

       }            

       if( t > 0x3fffffff)

       {

              buf[x] = t;

       }

}
```
编译执行example-6a.cpp得到如下结果。
<table><tr><td bgcolor=“#707070”>
kernel_time (hipEventElapsedTime) =442.050ms

mem_read_latency_cycle =   647 cycles for Vega10--1.536GHz
</td></tr></table>
使用extractkernel工具产生GCN assembly得到以下指令序列做一次显存读操作，总计5条VALU和1条SALU指令，这六条指令需要至少24个时钟周期， v_lshlrev_b64可能需要16个始终周期，那么可以得出显存读操作的延时为610个始终周期。
<table><tr><td bgcolor=“#707070”>
              v_mul_lo_u32 v2, v2, s3                                    // 000000001504: D2850002 00000702

              s_add_i32 s3, s2, -2                                       // 00000000150C: 8103C202

              v_ashrrev_i32_e32 v3, 31, v2                               // 000000001510: 2206049F

              v_lshlrev_b64 v[2:3], 2, v[2:3]                            // 000000001514: D28F0002 00020482

              v_add_co_u32_e32 v2, vcc, s0, v2                           // 00000000151C: 32040400

              v_addc_co_u32_e32 v3, vcc, v4, v3, vcc                     // 000000001520: 38060704

              global_load_dword v2, v[2:3], off                          // 000000001524: DC508000 027F0002

              s_waitcnt vmcnt(0)  
</td></tr></table>
2.6.2 CacheLine Length: 缓存行长度
-----------------------------

本节给出一个不太准确的测量缓存行长度的办法。参考下面的程序，buf中所有的值都为固定值1，而却只有一个thread，所有的读取地址都依赖于上一个地址，如果多个连续的读在同一个地址内，缓存产生命中，那么它的平均单笔延迟远小于从读显存延迟，否则非常接近读显存延迟。
```
__global__ void

test_kernel(hipLaunchParm lp,

       int* __restrict__ buf, int rangesize, int totalreads)

{

    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x; 

    int t = buf[x];

    //dependency reads

    for( int i=1; i < totalreads; i++)

    {

       int address = i * t * rangesize;

       address = address - 1;

       address = (address & (rangesize - 1)) | (address & (~(rangesize-1)));

       t = buf[address];

    }               

     if( t > 0x3fffffff)

     {

         buf[x] = t;

     }

}
```
编译执行example-6b.cpp得到如下输出结果，可以得出结论 Cacheline长度为64字节。
<table><tr><td bgcolor=“#707070”>
RangeSize[      16], kernel_time (hipEventElapsedTime) =4639.969ms

RangeSize[      16], mem_read_latency_cycle =   361 cycles for Vega10--1.536GHz

RangeSize[      32], kernel_time (hipEventElapsedTime) =3060.621ms

RangeSize[      32], mem_read_latency_cycle =   476 cycles for Vega10--1.536GHz

RangeSize[      64], kernel_time (hipEventElapsedTime) =2192.251ms

RangeSize[      64], mem_read_latency_cycle =   682 cycles for Vega10--1.536GHz

RangeSize[     128], kernel_time (hipEventElapsedTime) =1093.262ms

RangeSize[     128], mem_read_latency_cycle =   681 cycles for Vega10--1.536GHz

RangeSize[     256], kernel_time (hipEventElapsedTime) =566.791ms

RangeSize[     256], mem_read_latency_cycle =   706 cycles for Vega10--1.536GHz
</td></tr></table>
### 2.6.3 L1/L2 Cacheline Hit Latency：一/二级缓存命中延时

Example-6c.cpp展示一个简单的Kernel测量一级缓存命中的延时。设置rangesize = 1024，4096字节远小于16KB L2 Cache，那么L1 Cache的命中率接近99%。 将步长设置为Cacheline大小16DWORDs==64字节，那么每次读取指令都会指向一个新的Cacheline。
```
__global__ void

test_kernel(hipLaunchParm lp,

       int* __restrict__ buf, int rangesize, int totalreads)

{

    int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x; 

    int t = buf[x];

    //dependency reads

    for( int i=1; i < totalreads; i++)

    {

        int address = i * t * rangesize;

        address = address - 1;

        address = (address & (rangesize - 1));

        t = buf[address];

    }               

       if( t > 0x3fffffff)

       {

              buf[x] = t;

       }

}
```
编译执行example-6c.cpp 得到如下结果：
<table><tr><td bgcolor=“#707070”>
RangeSize[    4096], kernel_time (hipEventElapsedTime) =48.065ms

RangeSize[    4096], mem_read_latency_cycle =   239 cycles for Vega10--1.536GHz
</td></tr></table>
那么可以猜测L1 Cache命中延时小于239个时钟周期，用”extractkernel -i example-6c.exe”查看GCN Assembly 代码，获得主循环体代码如下：
<table><tr><td bgcolor=“#707070”>
0000000000001170 BB0_2:

        s_waitcnt vmcnt(0)                                        

        v_mul_lo_u32 v2, v2, s2                                   

        v_mov_b32_e32 v4, s1                                      

        v_mul_lo_u32 v2, v2, s5                                   

        s_add_i32 s5, s5, 1                                       

        s_cmp_lg_u32 s3, s5                                       

        v_add_u32_e32 v2, -1, v2                                  

        v_and_b32_e32 v2, s4, v2                                  

        v_ashrrev_i32_e32 v3, 31, v2                             

        v_lshlrev_b64 v[2:3], 2, v[2:3]                          

        v_add_co_u32_e32 v2, vcc, s0, v2                         

        v_addc_co_u32_e32 v3, vcc, v4, v3, vcc                   

        global_load_dword v2, v[2:3], off                        

        s_cbranch_scc1 BB0_2                                     
</td></tr></table>
GCN Assembly代码总计9条VALU指令， 4条Scalar指令，这些指令的延时需要64时钟周期，考虑到由于Cacheline不对齐会损失32-60个始终周期，L1 Cache命中的延时最低100个时钟周期，最高130个时钟周期。

Example-6d.cpp将rangesize修改为32768（128KB），编译执行获得如下结果。根据example-6c的分析，L2 CacheLIne命中的延时介于270-300个时钟周期之间。
<table><tr><td bgcolor=“#707070”>
RangeSize[  131072], kernel_time (hipEventElapsedTime) =75.581ms

RangeSize[  131072], mem_read_latency_cycle =   376 cycles for Vega10--1.536GHz
</td></tr></table>
2.7 Alternative Method to measure CacheLine Size：另一组测试Cacheline长度
-----------------------------------------------------------------

### 2.7.1测试CacheLine大小

Example-7a.cpp和example-7b.cpp尝试不断增加读写步长来Cacheline大小，该组测试已经被2.6.2代替。

### 2.7.2 Divergence for Memory Read/Write：显存访问分歧

Example-7c.cpp专门设计一个非常简单的方法产生显存读写分歧而导致的性能下降一半。让Thread0的显存地址计算和其他64个地址计算不同，这样编译器是否会产生两个不同global_store_dword指令，编译后检查Extractkernel产生的GCN assembly 代码，发现只有一条global_store_dword，对于这个简单的代码，HIPCC编译器表现良好。
```
__global__ void

test_kernel(hipLaunchParm lp,

       int* __restrict__ buf, int divergence )

{

       int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;        

       if ((hipThreadIdx_x & divergence) == divergence)

       {

               buf[x] = x;

       }

       else  

       {     

              buf[x&(NUM-1)] = x;

       }   

}
```
2.8   NCHW-4D Index Generation: 4D数组索引生成
----------------------------------------

在优化CNN卷积运算中，需要实时生成索引进行加速。假设我们需要生成NCHW对应Channel=0时候NHW个元素的索引。下面是简单代码实现，BlockDim = Dim3(256,1,1)， Grim = Dim3(H * W/256, N, 1)。
```
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
```
编译example-8a.cpp执行获得309GB/s的速度。考虑到hipLaunchKernel需要7微秒的额外开销，达到378GB/s的速度。考虑到数量比较小，相对于480GB/s的峰值性能，已经是很好的就成绩。
<table><tr><td bgcolor=“#707070”>
N*H*W=[1024,56,56], hipEventElapsedTime =38.715 microseconds, 309.001966 GB/s
</td></tr></table>
2.9 Local Data Share：本地数据共享
---------------------------

### 2.9.1 LDS Latency

GCN架构中LDS访问也是异步指令， 同显存读写指令一样，我们首先要获得LDS指令的延时。同理，使用一个线程，使用循环不断访问同一个地址，那么我们就可以获得LDS Latency。Mask防止访问越界， Thread0的Temp始终等于0， 该Mask并无特殊必要。
```
__global__ void

test_kernel(hipLaunchParm lp,

       int* __restrict__ buf, int mask, int outerLloops)

{

       int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;

       __shared__ int ldsData[4096];

       ldsData[hipThreadIdx_x] = buf[x];

       int temp = hipThreadIdx_x;

       for(int i = 0; i < outerLloops; i++){

              for(int j = 0; j < INNER_LOOPS; j++)

              {

                      temp = ldsData[temp] & mask;

              }

       }

       if (temp > 0)

       {

              buf[x] = temp;

       }

}
```
编译后example.cpp并使用extractkernel发现LDS read由如下序列指令：
<table><tr><td bgcolor=“#707070”>
              v_and_b32_e32 v0, s0, v0                    

              v_lshlrev_b32_e32 v0, 2, v0                 

              ds_read_b32 v0, v0                               

              s_waitcnt lgkmcnt(0)                             
</td></tr></table>
2条VALU指令需要20个时钟周期。执行example-9a获得如下结果，我们可以断定LDS 延时最好情况低于44个时钟周期：
<table><tr><td bgcolor=“#707070”>
latency for Vega10(1.536Ghz):  63 cycles
</td></tr></table>
### 2.9.2 LDS bank Conflicts

有32个Bank，如果每32threads中两个以上访问同一Bank，那么将造成Bank冲突，需要增加一个时钟周期来访问相同Bank的数据。下面的实例Buf的数据被初始化为和每个线程的hipThreadIdx_x相同，通过Stride来控制是否发生冲突，例如stride=1那么就是没有Bank冲突发生，否则有可能发生不同的Bank 冲突。

该实例只使用了64个threads即一个Wave，需要通过一个循环对4096个LDS单元做初始化。然后通过mask保证访问地址不越界。
```
__global__ void

test_kernel(hipLaunchParm lp,

       int* __restrict__ buf, int stride, int mask, int outerLloops)

{

       int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;

       __shared__ int ldsData[4096];

       for (int i = 0; i < NUM; i += 64)

       {

              ldsData[hipThreadIdx_x + i] = buf[hipThreadIdx_x + i];

       }

       int temp = (hipThreadIdx_x * stride) & mask;

       for(int i = 0; i < outerLloops; i++)

       {

              for(int j = 0; j < INNER_LOOPS; j++)

              {

                      temp = ((ldsData[temp] + hipThreadIdx_x)*stride ) & mask;

              }

       }

       if (temp > 0)

       {

              buf[x] = temp;

       }

}
```
按照惯例编译并执行example-9b.cpp，截取部分输出结果如下：
<table><tr><td bgcolor=“#707070”>
strdie = [1], latency for Vega10(1.536Ghz):  87 cycles

strdie = [2], latency for Vega10(1.536Ghz):  90 cycles

strdie = [3], latency for Vega10(1.536Ghz):  87 cycles

strdie = [4], latency for Vega10(1.536Ghz):  93 cycles

strdie = [5], latency for Vega10(1.536Ghz):  87 cycles

strdie = [6], latency for Vega10(1.536Ghz):  87 cycles

strdie = [7], latency for Vega10(1.536Ghz):  85 cycles

strdie = [8], latency for Vega10(1.536Ghz):  99 cycles

strdie = [9], latency for Vega10(1.536Ghz):  85 cycles

strdie = [10], latency for Vega10(1.536Ghz):  87 cycles

strdie = [11], latency for Vega10(1.536Ghz):  87 cycles

strdie = [12], latency for Vega10(1.536Ghz):  91 cycles

strdie = [13], latency for Vega10(1.536Ghz):  87 cycles

strdie = [14], latency for Vega10(1.536Ghz):  89 cycles

strdie = [15], latency for Vega10(1.536Ghz):  87 cycles

strdie = [16], latency for Vega10(1.536Ghz):  115 cycles
</td></tr></table>
结果非常有趣，Stride为奇数的延迟都为87Cycles以下， Stride=2, 4, 8,16的延迟急剧增加，stride为偶数的延迟大部分超过87 cycles，这和我们在其他文章中看到的一致，Stride为奇数能够消除Bank Conflicts，最糟糕的情况是Sttride= 2^N。

可以采用另外一个方法证明这个问题，做一个Excel表格，第一列依次为Thread ID 0-255，第二列为对应Stride=1的地址 == ThreadID * Stride, 第三列为对应的Bank ID =  (ThreadID * Stride) % 32，变换Stride，看看是否Bank ID能够均匀分布在0-31，如不能，则发生Bank  Conflicts。

2.10 Memory Channel Conflicts：存储通道冲突
------------------------------------

高端GPU都是基于多通道内存来提高带宽，那么每个通道的内存只能读写特定的地址空间。假设一个多通道显存设计，每4KB内存空间，分配给16个显存通道，那么每个显存通道只能读写其中的256字节的连续地址段。

下面的实例程序使用Proctectbits将保持高于16KB的地址不变，ShrinkBits将低位地址空间现在一个或者多个显存通道，那么将产生冲突，从而导致性能下降。
```
#define PROTECT_BITS  (0xFFFF0000)

__global__ void

test_kernel(hipLaunchParm lp,

       int* __restrict__ buf, int protectBits, int shrinkBits)

{

       int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;

       int address;

       address = (x & protectBits) | (x & shrinkBits);

       buf[address] = x;

}
```
我们编译执行example-10a.cpp获得下面结果，可以清楚看到最坏情况只有25%左右的性能。
<table><tr><td bgcolor=“#707070”>
Shrink Size in Bytes[128], bandwidth 181 (GB/S)

Shrink Size in Bytes[256], bandwidth 90 (GB/S)

Shrink Size in Bytes[512], bandwidth 181 (GB/S)

Shrink Size in Bytes[1024], bandwidth 360 (GB/S)

Shrink Size in Bytes[2048], bandwidth 359 (GB/S)

Shrink Size in Bytes[4096], bandwidth 359 (GB/S)

Shrink Size in Bytes[8192], bandwidth 359 (GB/S)

Shrink Size in Bytes[16384], bandwidth 359 (GB/S)

Shrink Size in Bytes[32768], bandwidth 359 (GB/S)

Shrink Size in Bytes[65536], bandwidth 359 (GB/S)

Shrink Size in Bytes[131072], bandwidth 358 (GB/S)
</td></tr></table>
例如SGEMM（单精度浮点矩阵乘法），如果矩阵A 的尺寸为 [4096, 4096]，矩阵B的尺寸也为[4096,4096]，那么读取矩阵A和矩阵B就会遇到存储通道读写冲突。

如果大范围测试M=N=K情况下的性能，从128开始，步长为16，会看到许多性能下降的组合，其中一个重要原因就是存储通道读写冲突引起。

SGEMM避免读写冲突的一个简单方法是使用Padding，例如K=4096，修改行的长度为4096+16，每行最后16个数据无效，可以有效提高性能。

2.11 Math Functions：数学函数
------------------------

如果对CPU的数学函数做过测试，都应该知道每条数学函数需要数十到数百条指令完成。数学函数在计算机中使用最低六次泰勒级数展开，加上额外的一些操作，数十条指令是非常正常的。每下面一个实例用双精度(Double Precision)三角函数来测试数学。
```
#define INNER_LOOPS  100

#define OUTER_LOOPS  10000

__global__ void

test_kernel(hipLaunchParm lp,

       int* __restrict__ buf, int outerLoops)

{

       int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;

       double f = sin(x / 256.0);

       for (int i = 0; i < outerLoops; i++)

              for (int j = 0; j < INNER_LOOPS;j++)

                      f = sin(f);

       if (f > 0.999)

       {

              buf[x] = f;

       }

}
```
编译执行example-11a.cpp得到如下结果：
<table><tr><td bgcolor=“#707070”>
sin --double needs 2339 cycles
</td></tr></table>
该结果符合预期， sin的数学函数实现分两个部分，把角度映射到[0, 2Pi]，将耗费大量指令，然后使用泰勒级数展开，同时Mi25的FMA64只有1/16的速度，双精度Sin超过了140条指令。有兴趣的可以尝试单精度sin, cos, log, exp,  tan,  arcsin, sqrt, rsqrt等常用超越函数的开销。

基础的数学定理可以 大大减少计算开销，例如 exp(x, y) * exp(x,z) 等价于 exp(x, y + z),   if  (sqrt(a) < b) 等价于  if ( a < b *b)， if ( arcsin(a)  < arcsin(b)) 等价于 if  ( a < b)。

2.12 Reduction：归约
-----------------

Reduction是一个非常常见的操作，例如求一个数组的最大、最小值，或者求和。常见的GPU实现，第一步将所有数据写到LDS，第二步有效Threads减半，每个有效线程读两个数，求和，然后结果写回LDS，重复步骤二直到有效线程数为1。根据我们前面的测试，LDS读写的延迟比较大， 如果每次对4个数求和，是否可以大大提高读写速度？
```
__global__ void

test_kernel(hipLaunchParm lp,

        int* __restrict__ buf, int reduce_number_once)

{

        int x = hipBlockDim_x * hipBlockIdx_x + hipThreadIdx_x;

        __shared__ int ldsData[256];

        ldsData[hipThreadIdx_x] = buf[x];

        __syncthreads();

        int sum =0;

        if (reduce_number_once == 2)

        {

                for (int s = 256 >> 1; s > 0; s = s >> 1)

                {

                        if (s > hipThreadIdx_x) {

                                ldsData[hipThreadIdx_x] = ldsData[hipThreadIdx_x] +

                                                           ldsData[hipThreadIdx_x + s];

                        }

                        __syncthreads();

                }

                if (hipThreadIdx_x == 0)

                {

                        sum += ldsData[0];

                }

        }

        if (reduce_number_once == 4)

        {

            for (int s = 256 >> 2; s > 0; s = s >> 2)

            {

               if (s > hipThreadIdx_x) {

                    ldsData[hipThreadIdx_x] =  ldsData[hipThreadIdx_x] +

                                               ldsData[hipThreadIdx_x + s] +

                                              ldsData[hipThreadIdx_x + 2 * s] +

                                               ldsData[hipThreadIdx_x + 3 * s];

                }

            }

            if (hipThreadIdx_x == 0)

            {

               sum += ldsData[0];

            }

        }

        if ((hipThreadIdx_x == 0) && sum > 9999)

        {

                buf[hipBlockIdx_x] = sum;

        }

}
```
编译执行example-12a.cpp得到如下结果：
<table><tr><td bgcolor=“#707070”>
Reduce 2 once:  elapsed time:4.80159

Reduce 4 once:  elapsed time:2.817486
</td></tr></table>
每次读4个LDS数据比每次读2两个数据性能提高了70%。Reduction可以看作是LDS读写延迟为主要性能瓶颈， 减少程序需要等待的LDS读写延迟将大大提高程序性能。 如果每次读8个LDS数据，并对八个数据求和，那么需要8*8*8=512个元素。读者可以自己尝试是否可以进一步提高性能。

2.13  Padding Before Convolution
--------------------------------

### 2.13.1 1st Padding Kernel

在CNN的Convolution，如果Filter Size大于1x1，那么Padding(填充)是一个非常重要的函数。假设BatchSize=1024, Channels=1024， Height=Width=7, Padding=3X3，那么Padding之后的Height=Width=13x13，13x13=169远远小于256，因此我们需要每个Threads读写超过一个Channel的数据。下面的代码BlockDim=Dim3(256,1,1)，GridDim= (【13 * 13/256】,  Channeles=1024, BatchSize=1024)。代码先计算输入原始输入数据的地址，如果在 【7，7】的范围内，那么需要读取显存数据，否则设置为Padding Value== 0.
```
__global__ void

test_kernel(hipLaunchParm lp,

       float* __restrict__ bufA, float* __restrict__ bufB, int channels_once, int c, int h, int w,  int padding )

{

       int hw =  hipThreadIdx_x;

       int cc = channels_once * hipBlockIdx_y;

       int n = hipBlockIdx_z;

       float org_data[16];

       if (hw < (h * w))

       {

              int hh = hw / w - padding;

              int ww = hw % w - padding ;

              for (int i = 0; i < 16; i++)

              {

                      org_data[i] = 0.0f;

              }

              int in_w = w - 2 * padding;

              int in_h = h - 2 * padding;

              bool needFetching = (ww >=0) && (ww < (in_w)) && (hh >= 0) &&

                                 (hh < (in_h));

              if (needFetching == true) {

                      int base = n * c * in_h * in_w + cc * in_h * in_w +

                            hh * in_w + ww;

                      for (int i = 0; i < channels_once; i++)

                      {

                             org_data[i] = bufA[base + i * in_h * in_w];

                      }

              }

              int base = n * c * h * w + cc * h * w + hw;

              for (int i = 0; i < channels_once; i++)

              {

                      bufB[base + i * h * w] = org_data[i];

              }

       }

}     
```
编译并执行example-13a.cpp。得到如下输出结果：
<table><tr><td bgcolor=“#707070”>
Read/Write [1] Channels per thread:  elapsed time:29.635487

Read/Write [1] Channels per thread:  ==> Estimated Bandwidth 44  GB/s

Read/Write [2] Channels per thread:  elapsed time:21.011665

Read/Write [2] Channels per thread:  ==> Estimated Bandwidth 62  GB/s

Read/Write [4] Channels per thread:  elapsed time:14.498355

Read/Write [4] Channels per thread:  ==> Estimated Bandwidth 91  GB/s

Read/Write [8] Channels per thread:  elapsed time:11.157874

Read/Write [8] Channels per thread:  ==> Estimated Bandwidth 118  GB/s

Read/Write [16] Channels per thread:  elapsed time:9.165571

Read/Write [16] Channels per thread:  ==> Estimated Bandwidth 144  GB/s
</td></tr></table>
获得的性能非常低，远远低于480 GB/s的理论极限。 使用”extractkernels example-13.exe”获得编译后的GCN汇编程序， 发现以下奇怪代码，总共包含16条buffer_store_dword，和一条buffer_load_dword值令。
<table><tr><td bgcolor=“#707070”>
               v_mov_b32_e32 v4, 0

              buffer_store_dword v4, off, s[0:3], s11 offset:64

              buffer_store_dword v4, off, s[0:3], s11 offset:56

              buffer_store_dword v4, off, s[0:3], s11 offset:48

              buffer_store_dword v4, off, s[0:3], s11 offset:44

              buffer_store_dword v4, off, s[0:3], s11 offset:36

              buffer_store_dword v4, off, s[0:3], s11 offset:32

              buffer_store_dword v4, off, s[0:3], s11 offset:20

              buffer_store_dword v4, off, s[0:3], s11 offset:16

              buffer_store_dword v4, off, s[0:3], s11 offset:8 

              buffer_store_dword v4, off, s[0:3], s11 offset:60

              buffer_store_dword v4, off, s[0:3], s11 offset:52

              buffer_store_dword v4, off, s[0:3], s11 offset:40

              buffer_store_dword v4, off, s[0:3], s11 offset:28

              buffer_store_dword v4, off, s[0:3], s11 offset:24

              buffer_store_dword v4, off, s[0:3], s11 offset:12

              buffer_store_dword v4, off, s[0:3], s11 offset:4 

              …

              buffer_store_dword v4, v2, s[0:3], s11 offen

              buffer_load_dword v6, v2, s[0:3], s11 offen
</td></tr></table>
而同时我们从以前的经验获知，HIPCC编译器通常使用global_load_dword和global_store_dwor指令读写显存数据。16条写显存指令和程序中初始化”org_data[i] =0.0f”最接近，为证实这个猜测修改为”org_data[i] =0.1111f”，“v_mov_b32_e32 v4, 0”变成了“v_mov_b32_e32 v4, 0x3de38da4”。编译器在16个org_data的初始化为0后，然后把org_data缓存到显存，然后使用时再从显存读出，这样程序的效率大大降低。 通常只有在寄存器超过256时，编译器才需要使用显存补充缺失的存储器。这个简单程序显然不需要这么多寄存器。HIPCC编译器把这块显存称为scratch（参考产生的GCN汇编程序中的scratch_hi 和 scratch_lo）。

         一个可能的猜测是循环变量channles_once作为输入参数出现，而编译器无法判别总的循环次数，不能判别需要org_data的实际大小，而把导致org_data被分配到scratch memory。

### 2.13.2 Optimize Kernel to Remove Scratch Memory

Example-13b.cpp把所有的整数参数转为了常量，已尝试是否会消除scratch memory。

编译并测试example-13b.cpp得到如下结果：
<table><tr><td bgcolor=“#707070”>
Read/Write [16] Channels per thread:  elapsed time:2.929695

Read/Write [16] Channels per thread:  ==> Estimated Bandwidth 450  GB/s
</td></tr></table>
本实例的每个线程读写16个Channels，完全有可能减低到4个Channels也能获得非常接近的性能。读者可以试一试。另外，读者也可以尝试读取1，2，8个Channels的不同性能。

2.14 BatchNorm
--------------

BatchNorm的基本原理参考： [https://blog.csdn.net/hjimce/article/details/50866313](https://blog.csdn.net/hjimce/article/details/50866313)

根据基本原理，最简单的实现需要读取每个元素三次，第一次是计算平均值，第二次是计算平均方差，第三次是计算BN值，每次存储读取失败需要重新向L2请求数据，这样无法获得最佳性能。GCN架构的L1 Cache 总共有256 Cachelines ( 16 KB /64 Bytes per CacheLine)，如果有256个像素，BatchSize大于16，那么需要读取的Cacheline将超过256。平均方差和平均值可以用同一个循环完成，这样可以减少一次L1 Cache的数据读取。再进一步，如果读取的数据能够保存在VGPR中，那么仅仅读取一次L1 Cache即可。总共设计了四个测试：

*   Example-14a.cpp：使用了三次L1 Cache读写的方式，性能为22G Pixels/s。
*   Example-14b.cpp：使用了一次L1 Cache 读写，将128个Batch的数据保存在2个Threads中，性能为15 G Pixels/s。
*   Example-14c.cpp：使用了一次L1 Cache 读写，将128个Batch的数据保存在4个Threads中，性能为32G Pixels/s。
*   Example-14d.cpp：使用了两次L1 Cache 读写，第一次读L1 Cache计算平均方差和平均值，第二次读L1 Cache做(L1/L2可能是命中失败)，性能为30G Pixels/s。

理论上方法14b和14c应该取得一样的性能，因为这两个方法仅仅读取一次L1 Cache，而且需要的VPGR数都是小于80。而实际测试的结果完全不符合预期，方法14b和14c应该远远高于方法14d。基本的猜测是HIPCC编译器有不为人知的特性。使用extractkernels工具产生 GCN assembly代码，并进行分析：

*   Example-14a.cpp：产生的代码极为简单，使用的VGPR数量低于16个；
*   Example-14b.cpp：产生的代码非常复杂，VGPR达到了最大值255，而且使用scratch memory来替代不足的VPGRs；
*   Example-14c.cpp：代码比较复杂， 使用超过105个VGPR，低于128个VGPR，没有使用scratch memory；
*   Example-14d.cpp：产生的代码极为简单，使用的VGPR数量低于16个；
*   所有四个实例中计算显存地址部分没有任何优化，浪费了大量计算指令；

HIPCC的寄存器分配和显存地址计算的性能较差，在本例中无法获得最佳性能，如需要获得最佳性能，需要用汇编代码进行优化。

3     其他
========

Miopen提供了大量实例使用汇编指令提高性能，可以作为参考。[https://github.com/adityaatluri/gemm-vega64](https://github.com/adityaatluri/gemm-vega64)提供了inline assembly的方式简化GCN架构的HIP/OpenCL汇编程序，可以作为极好的参考。

4 Convert Word to MarkDown
==========================

WORD to MD file

把WORD文件内容放入下面网站， 转换为HTML

[https://wordhtml.com/](https://wordhtml.com/)

然后把HTML内容通过另外一个网站转换为MarkDown

[https://tool.lu/markdown/](https://tool.lu/markdown/)Contents
