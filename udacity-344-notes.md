# Udacity 344 Intro to Parallel Programming

- 本课程除了教你在GPU上进行并行计算外，重要的是介绍并行计算的思维

## Lesson 1 GPU Programming Model

### CPU vs. GPU 架构

- 如何让处理器处理的更有效率：
  - 提高时钟频率，让单位时间执行的指令更多
  - 提高单条指令的能力，比如处理更多的数据，SIMD
  - 同时有多个执行单元，多核
- 选择2头牛还是选择1024只小鸡，在并行计算中，往往会选择1024只小鸡
- GPGPU: 图片处理单元（GPU）上的通用编程
- 这些年晶体管越来越小，已经达到了5nm以下，我们可以在同样尺寸上芯片上放更多的晶体管来达到更大的算力，但是为什么这些年CPU芯片的时钟频率没有怎么提高？首先是功耗，其次是散热。[为什么主流CPU的频率止步于4G?](https://zhuanlan.zhihu.com/p/30409360)
- CPU架构上花了很多晶体管在电力与逻辑控制上，所以计算的效率并非最优的，GPU是有优势的。
- 传统CPU优化的是单条指令的执行时间（latency，延时），而GPU优化的是吞吐量（throughput）。

### GPU编程

- GPU像是一个CPU的协处理器，GPU是自己的存储，我们一般称为显存。

- 程序的执行是由CPU发起的，在整个程序运行过程中，可以通过CPU调用GPU的能力，比如驱动GPU申请一块显存、驱动GPU完成显存往内存上的拷贝、 驱动GPU执行一个kernel函数等。

- GPU编程的经典步骤
  - 用cudaMalloc申请显存
  - 用cudaMemCpy把内存数据拷贝到显存上
  - 执行Kernel函数，并行的对这些数据执行计算
  - 将结果cudaMemCpy回内存上
  
- GPU上编程最核心的就是如何编程kernel函数，它的代码就像只在一个核上运行。

- GPU非常擅长做2件事情：1）同时启动大量的线程； 2）同时运行多个线程 。

- kernel函数调用的写法`kernel_function<<<dim3(bx,by,bz), dim3(tx,ty,tz),shmem>>>(args...)`。

- 其中dim3是一个数据结构，有`x`、`y`、`z` 3个成员。隐藏转换`num`为`dim3(num, 0, 0)`。

- `threadIdx`用于在kernel函数中获取当前线程在`block`中的定位，`blockIdx`用于获取当前`block`在`grid`中的定位。

- 在kernel函数中通过`blockDim`来获取一个`block`的维度，通过`gridDim`来获取`grid`的维度。

- 每个GPU都有一些编程限制，可以通过`deviceQuery`程序来查看：

  ```txt
  Total amount of global memory:                 6075 MBytes (6370295808 bytes)
  (10) Multiprocessors, (128) CUDA Cores/MP:     1280 CUDA Cores
  GPU Max Clock rate:                            1709 MHz (1.71 GHz)
  Memory Clock rate:                             4004 Mhz
  Memory Bus Width:                              192-bit
  L2 Cache Size:                                 1572864 bytes
  Total amount of constant memory:               65536 bytes
  Total amount of shared memory per block:       49152 bytes
  Total number of registers available per block: 65536
  Warp size:                                     32
  Maximum number of threads per multiprocessor:  2048
  Maximum number of threads per block:           1024
  Max dimension size of a thread block (x,y,z): (1024, 1024, 64)
  Max dimension size of a grid size    (x,y,z): (2147483647, 65535, 65535)
  Maximum memory pitch:                          2147483647 byte
  ```
  
- 如上图所示，每个block中最大有1024个线程，而每个sm中最大有2048个线程，所以每个sm最大同时执行2个block。

- 在我1060的卡上测试，如果blockDim设置为(1024, 2, 1)，使用`cudaGetLastError`获取错误返回为：`invalid configuration argument`。

##  Lesson 2 GPU Hardware and Parallel Communication Pattern

### 并计算中的通信模式

- Map: 每个对应的输出位置，访问其一一对应的输入位置
- Gather：每个输出位置，都会访问多个的输入位置
- Scatter：每个输入位置，都会对应被多个输出位置读取
- Stencil：每个输出位置，都访问输入中一个模板对应的位置
- Transpose：输入输出一一对应，但在2维结构上，是转置的

![Parallel Communication Patterns](./images/image-20210502160734412.png)

### GPU架构

-  我们通过kernel函数启动的大量GPU线程，是按块，划分为多组的。其中一个block里的所有线程，会调度到同一个流处理器（SM）上执行，每个SM上都有一个shared_memory，可以被这一组线程共享访问。
- 多个block被调度到不同的SM上，顺序是不能保证，执行时间也没有保证。
- 不同的kenrel函数之间执行是串行的，只当前的kenrel函数中的所有block都执行完后，下一个kernel函数才会执行。

![SM and Blocks](./images/image-20210502161240839.png)

- GPU的内存层次：每个线程都有自己的local memory，每个SM有shard memory，可以被一个block中的多个线程共享访问；所有SM中的线程都可以访问global memory。

- 同一个block中的线程经常需要共享操作shared_memory，比如先写再读，那就需要等所有线程都写完后，所有线程才能开始读。这时就需要所有线程同步，`__syncthreads()`就是一种`barrier`措施，在插入了`__syncthreads()`的地方，所有线程执行到这里时，都需要等待，当前所线程都到达时，才再次开始同时执行。

- 对于shared memory的使用，一般都是先写入，后用；可以用在那些一次载入，多次使用的数据上。如果访问模式中有错位（对应线程序号，访问了非对应位置的shared memory)，那都需要在读取后，设置`barrier`，然后读取。比如：

  ```c
  __shared__ int arr[128];
  // arr[i] = arr[i + 1];
  temp = arr[i + 1];
  __synthreads();
  arr[i] = temp;
  ```

### 编写高性CUDA程序

- 最大化计算密度，计算密度 = 计算 / 访存
- CUDA程序优化的核心往往都是优化访存的时间
- 对于Gobal Memory的访问，特别要注意，连续访问的性能，比按Stride的性能高，更比随机访问的性能要高。（这里应该是因为有L2 Cache的原因）
- 多线程访问同一个global memory资源时，需要加锁； 或使用原子函数：`atomicAdd`等。
- 避免线程发散（不同的线程，走了不同的if-else分支，或循环次数不同）。

## Lesson 3 Fundamental GPU Algorithms: Reduce、SCAN、Histogram

- GPU算法中，往往通过度量`Step`和`work`来评估并行算法的复杂度
- Reduce算法的定义：
  - Set of Elements
  - Reduce Operator: 1）2元运算法； 2）具有结合性 (a op b) op c  = a op (b op c)
- Reduce算法的GPU实现：
  - 按线程块划分，每个线程块分别Reduce，然后把所有线程块的结果再Reduce一下。
  - 每个线程块内采用步长的循环，两两相加，结果写在前一个结果上。
  - Reduce在GPU实现的STEP复杂度为`O(logN)`，WORK复杂度为`O(N)`
- SCAN算法的定义
  - Set of Elements
  - Reduce Operator: 1）2元运算法； 2）具有结合性 (a op b) op c  = a op (b op c)
  - 存在单位元I，满足 a op I = a
- SCAN算法包括2种形式：1）Inclusive； 2）Exclusive
- Hillis Steele SCAN算法
- Blelloch SCAN算法
- Histogram算法

## Lesson 4 Fundamental GPU Algorithm: Application of Scan and Sort

- Compression

- Merge Sort
- Bitonic Sort
- Radix Sort

## Lesson 5 Optimizing GPU Programs

- 高效GPU程序的几个原则
  - 增大程序的计算密度
  - 减小程序的存储操作密度
  - 合并全局内存访问 
  - 避免线程发散
  - 利用好存储的层次结构
- 优化的层次，一般注重前3条就行了，在GPU下，后2条，尤其是最后一条的收益不大
  - 选择好的算法
  - 遵守GPU编程的基本原则（上面那几条）
  - 与架构相关的详细优化
  - 指令级别的微优化
- APOD流程
  - Analyze: 部析程序的热点部分，是不是可以并行化加速
  - Parallelize：使用AVX、OpenACC、OpenMP、CUDA等来加速
  - Optimize
  - Deploy：在真实的数据与环境上运行，看效果
- 矩阵转置的GPU实现
  - 直接GPU加速会，导致DRAM的带宽使用率低，因为会出来大Stride写入的问题
  - 每次Load一块，在shared memory中进行转置，再拷贝回去
  - 使用shard memory就会遇到 __syncthreads的问题，这时候，需要适当降低线程块中线程的数量
- 优化所有线程在`barrier`前面的等待时间
- 避免同一个`wrap`中所有线程的线程发散（有不同的执行分支），因为所有wrap中的线程是同步执行的，如果有分支，那当有一部线程在执行某个分支时，其他线程在等待。
- 使用一些内置的数学函数：`__sin`、`__cos`、`__exp()`等
- 使用`pinned memory`
- 使用MultiStream和异步拷贝。不同stream上内存操作与核函数执行都是可以异步的，同一个stream上的拷贝与核函数执行是排队的。

## Lesson 6 Parallel Computing Patterns

对于GPU编程来说，最大的挑战在于，对之前从来没有见过的问题或求解模式进行并行化。所以本节的主要内容，就是看一些有趣的问题，如何被并行化的。

### Dense N-Body

这个问题的背景与解决方法，可以参考NVIDIA的官方博客：[chapter-31-fast-n-body-simulation-cuda](https://developer.nvidia.com/gpugems/gpugems3/part-v-physics-simulation/chapter-31-fast-n-body-simulation-cuda)

All Pair N-Body是N-body中最简单的模式，也被称为蛮力方法，包括分开计算每对元素之间的力，然后对每个元素上产生的力进行加和。它的算法复杂度是$O(n^2)$。

对于Nbody问题，有很多可以加速的方法，用于近似计算，比如基于树的方法，有种算法叫Barnes Hut，它的算法复杂度是$O(n\log(n))$。

快速多极（Multi-pole）算法可以用于高精度的快速近似计算，复杂度为$O(n)$：它的本质是把遥远的物体混为一体，在计算过程中把它们视为一个单一的整体。快速多极算法实际上被评为20世纪10大算法之一。

multipole算法：[论文](https://aip.scitation.org/doi/pdf/10.1063/1.4773727)，[实现](https://github.com/duanebyer/nbody)

我们这里将通过演示All pair Nbody的计算问题来演示，如何有效地使用内存层次结构来加速我们的计算过程。

朴素来实现的话，我们都使用global memory，那么在我们计算$N\times N$的Nbody的关系矩阵时，每个元素都需要被我们反复多次的读入内存，如果我们是使用$N^2$的线程，每个线程计算一对pair之间的力，那每个元素会被从global memory中加载2N次，分别是该序列在$N\times N$矩阵中对应的那一行和那一列。

![N-Body Tiling](./images/image-20211013200232534.png)

将整个矩阵划分为网格，每个网格我们称之为一个Tile，它的大小是$p\times p$，正常情况下，我们要计算这个网络内的矩阵值的话，我们需要把这个网格涉及到的元素从global memory加载，每个元素都要被加载$2p$次。

我们的优化目标是，对于一个Tile中需要加载的元素，我们只加载1次。

如果我们使用的方式是对于这个Tile中的$p^2$个点，使用$p^2$个线程，那会存在很多的问题：1）这些线程之间需要通过sharedMemory来共享数据；2）当我们进行力的横向加和时，需要考虑线程之间的同步问题。

另外一种方式是是只使用$p$个线程，每个线程在横向上计算，使用一个`for`循环。

![Tile Calculation](./images/image-20211013200956911.png)

这种模式下，只需要将sourceParams加载为sharedMemory中，而dstParams不需要共享了。也不需要在不同线程间来汇总单独的力。

![p Thread Tiling](./images/image-20211013201208325.png)

这种模式降低了整个程序的并行度，但当我们解决的Nbody问题的N很大的时候，我们对线程的使用率是很高的，这种模式也能达到很高的计算吞吐。

选择多大的网格$p$，如果网格太大，会导致我们要加载进sharedMomory中的sourceParms较大，有可能sharedMemory放不下。同时如果网格太大，将会导致线程块较少，可能用不完所有的SM。

如果网格太小，那么对于global memory的访问压力就会比较大。

![small p vs big p](./images/image-20211013202132264.png)

 这个例子说明了，在一些问题上，适当让每个线程做更多的事，减少并发程度，可能会取得更好的效果。

### 稀疏向量乘法实现优化SpMV

![csr scalar kernel](./images/image-20211013203327645.png)

上面的代码计算的是$y += Mx$，其中$M$是稀疏矩阵，$x$和$y$都是列向量。

在cuda代码中，每个线程计算y的一行的结果。

由于M中每行的长度可能不一样，导致了一个线程束（wrapper）的执行时间，取决于运行最慢的那个线程。

这种情况下我们可以采用，每个线程只计算2个数乘法，然后用bacward inclusive SCAN SUM来计算每一行乘积的加和。这种算法对于每一行长度的变化就不敏感了。但这种算法需要线程间通信了。

![Elements Per Row](./images/image-20211013205004760.png)

针对这个问题，比较学术的讨论可以参考：[Efficient Sparse Matrix-Vector Multiplication on CUDA](http://wnbell.com/blog/2008/12/01/efficient-sparse-matrix-vector-multiplication-on-cuda/)

![Hybrid Approach](./images/image-20211013205241952.png)

这个例子说明了2个优化思路：

* 需要让线程尽可能的busy，一个wapper里的线程不能太发散
* 管理通信开销同样重要

### 图的广度优先搜索

对于一个图来说，如果每个顶点相连接的边很多，则这个图是一个稠密的图，相反，如果边很少，则是一个稀疏的图。

现实中的很多问题都可以用图模型来建模，比如整个Web可以看成一个超大的图，页面就是顶点，而页面之间的超连接关系构成了边；人的社交关系就是一个很大的图，顶点就是每个人，边则是两个人认识，互为好友关系。

对于图结构，最广泛的操作就是对图中所有的结点的遍历，比如对于Web，我们的爬虫程序实现就是在执行对于整个Web图网络的遍历。对于图的遍历一般有2种方法，深度优先（DFS：Depth First）和广义优先（BFS：Breadth Frist）。

![DFS vs BFS](./images/image-20211014191022374.png)

DFS算法需用的缓存会少一些，而BFS则有较高的并行度，但对中间存储要求高一些。

对于BFS来说，我们比较关心的是从root结点来使，遍历整个图的深度。对于n个结点的图来说，最大可能的深度为n-1（串连的形状），则小可能的深度为1（一个星状发散的形状）。

![Depth of Graph](./images/image-20211014191557469.png)

显示当图最有最小深度时，这个图的结构是最合适并行化遍历的。

接下来我们研究如何设计一个好的并行化算法来执行BFS。我们希望这个算法具有以下的特性：

* 并行度高
* 合并内存访问
* 最小化线程发散
* 容易实现

先来看第一种算法，这个算法的设计比较一般。它的思想，就是从一个入口点开始，一圈一圈的向外来设置每个顶点的度，每一圈就是一次迭代。

首先我们使用一个数组v[n]代表n个结点的深度，根节点的深度为0，初始状态下都为-1，我们可以用这个数组也样表示这个结点有没有被访问过。用一个pair来表示边，比如下面的图，我们可以表示为：

Vertices：0, 1, 2, 3, 4, 5, 6

Edges：（0,1), (1,2), (2,3), (3,4), (2,5), (5,6)

![image-20211014193708009](./images/image-20211014193708009.png)

那我们遍历的算法可以描述为以下：

* 对整个图进行多次迭代，迭代次数，取决于图的最大深度
* 每轮迭代，遍历所有的边，检查这条边的两个顶点的深度值v[first]和v[second]，如果有一个不为-1，另一个为-1，则将为-1的那个顶点的度设置为不为-1顶点的深度+1。
* 某轮迭代，所有的点的深度都不变化时，迭代结束。

初始化每个顶点的度，入口顶点初始化为0。

![image-20211014194113646](./images/image-20211014194113646.png)

bfs的kernel函数，并行的处理每条边，然后看左右结点的度的情况。

需要注意的是，这里可能会存在多个线程同时修改vertices的情况，按道理这里要加锁的，不知道代码里，为什么不加，课程中描述的是，可以忽略，因为重复赋值是没有关系的，因为多个线程只会赋值相同的值。

![image-20211014194524972](./images/image-20211014194524972.png)

结束迭代的代码逻辑，kernel代码中，如果有顶点的深度被修改了，就是把done设置为false，每一轮迭代kernel执行完，就会把done从device拷贝到host上。然后在再一轮迭代开始前，先设置为true。

![image-20211014194223840](./images/image-20211014194223840.png)

这个算法的并行度、内存访问、线程发散情况都不错，但是算法复杂度是O(VE)，复杂度较高。

这个算法显然有很多冗余的地方，对于边的访问，尤其是到了后面的迭代轮次，大部分边连接的顶点都已经被设置过有效的度了，但每一轮迭代都需要处理一遍。

**另外一种算法**：

该算法是由Duane Merill和他的同事发表于2012年。[Duane Merill](https://github.com/dumerrill)应该是cub库的核心贡献者。

如果我们能够有效识别出来，哪些是已经访问过的结点（设置为度），哪些是未访问的圈外结点，就好了。

![image-20211028200316357](./images/image-20211028200316357.png)

首先新的算法，将换一种图结构的表示算法，有些类似于稀疏矩阵CSR表示方法：

![image-20211028201246005](./images/image-20211028201246005.png)

其中的C数组，按顺保存每个结点的邻居的编号，比如1号结点的邻居是(0, 2, 4)，C的长度就是边的个数

而R数组长度是结点个数+1，它表示的是每个结点的邻居在C数组中的开始位置。可以看出第i个结点的邻居在C中就是[R[i], R[i+1])区间内。像2号结点，它没有邻居，它的邻居在C中的范围是[5,5)，是个空集合。

D数组表示的是每个顶点的度，它的长度就是顶点的个数，初始化为-1，入口点的度初始化为0。

![image-20211028201843279](./images/image-20211028201843279.png)

这个算法的运行步骤如下：

初始化：初始点的深度为0，初始化Fronter（前沿点）为初始化的邻居点。

1. 并行计算每个Fronter中的点v的邻居个数，即R[v+1] - R[v]
2. 为新的Fronter申请空间
3. 执行稀疏并发复制。
4. 在新的数组上执行并发访问，修改D数组，类似于前面一种算法。

比如对于算法的第一轮，Fronter为[1,3]，对应的邻居个数是R[2]-R[1]=3，R[4]-R[3]=1，所以一共需要新申请4个空间。然后使用SCAN来计算稀疏拷贝在新数组中的起点位置，即[0, 3]。分别将C[R[1]]开始的3条数据拷贝到新数组下标0开始的空间上，将C[R[3]]开始的1个数据拷贝到新数组下标3开始的空间上。

### List Ranking

![image-20211104183502068](./images/image-20211104183502068.png)

对于一个链表来说，每个结点都有一个指针指向了其后继结点。现在我们的需求是，从每个结点开始，要访问到链表的最后一个元素。

我们为每个结点增加一个指针为Chum，通过循环，每次将Chum向前移动，那最终都会移动到最后一个结点。

那么平均要循环n次，才能让最左边的结点的chum指针指向最后一个结点。

我们有更快的方法吗？

由Danny Hillis和Guy Steele在1986年提出了新算法（新算法并不是由他们发现，他们只是做了很好的描述）。

每轮迭代，本身我们只能向前前进一格，但通过指针的指针，我们可以向前前进很多。

![image-20211104184634679](./images/image-20211104184634679.png)

![image-20211104185028372](./images/image-20211104185028372.png)

正式引入List Ranking问题：

输入：每一个结点和其后续结点，并指定一个起始结点

输出：按顺输出每个结点

这个算法用CPU实现非常容易，从每一个结点开始，依次查找下去就可以了，复杂度是O(n)

![image-20211104185353342](./images/image-20211104185353342.png)

下面我们介绍在GPU上并行度较高的算法，它的算法复杂度是O(logn)，但它实际的Work是比CPU多的，但Step比CPU少。

这个算法分为2个阶段：

第一个阶段通过迭代来查询离每个结点一定hops的结点，每轮迭代，分别查找的是1,2,4,8这样的hops的结点。

第二个阶段需要计算出每个结点的在最终序列中的输出位置，也是需要logN次迭代，每次迭代会激活一部分结点（输出该部分结点的最终位置）

首先被激活的是Start结点，它输出在0号位置

0， -1， -1， -1， -1， -1， -1， -1， -1， -1

第一次迭代，按1hops来查找已经激活结点的后续，然后将其激活

0， -1， -1， -1， -1，  1， -1， -1， -1， -1

第二次迭代，按2hops来查找已经激活结点的后续，然后将其激活，被激活点的outpos为激活结点的outpos+2(hops)

0，-1， 2， -1， -1， 1， -1，  3， -1， -1

后续类推，4hops的

0，6， 2， -1， 4， 1， 7，  3， -1， 5

8hops的：

0，6， 2， 8， 4， 1， 7，  3， 9， 5



![image-20211104190153023](./images/image-20211104190153023.png)

### 并行算法在HashTable上的应用

在一般的HashTable中，假设每个桶中是通过链表或红黑树来解决冲突问题。

对于GPU并发的查找问题，由于每个线程查询的Key不一样，所以它们有的可能很快就查询到返回了，有的访问到一些特别大的桶中，返回很慢，容易出来Wrap发散问题。

而对于插入的话，更是需要解决数据冲突的问题，比如多个线程同时往HashTable中插入同样的Key。

结论就是，通过链表方法来解决冲突的HashTable对于GPU并发非常不友好。

这里提出的是布谷鸟（CucKoo）哈希算法，布谷鸟会把自己的蛋生在别的鸟窝中。

新算法的哈希表是有多个哈希表（每个哈希表一个HashFunction）组成，对于N个元素要往哈希表中添加，按顺序插入某一张表中，没有冲突，则插入到桶中，如果有冲突，则将原来桶中的元素挤出来，原来的元素，则向后一张哈希表中找位置，如果有冲突，则再把别人挤掉，直到最终都找到位置。

它的好处就是在GPU可以对多个哈希表同时计算HashFunction并进行查找。

## Lesson 7  Additional Parallel Computing Topics

### Parallel Optimization Pattern

7种并发编程模式：[Optimization and architecture effects on GPU](./papers/Optimization_and_architecture_effects_on_GPU_compu.pdf)

#### Data layout transformation

它的核心是重新整理数据的布局，以获取更好的内存性能。有一个叫Burst utilization的指标可以来衡量。

这里举的例子是AoS（Array of Structure）-> SoA （Structure of Array）

![image-20211108201314365](./images/image-20211108201314365.png)



![image-20211108201407186](./images/image-20211108201407186.png)

对于像上面这样的代码，SoA会更具有性能优势。

![image-20211108201541986](./images/image-20211108201541986.png)

#### Scatter-to-Gather transformation

Scatter和Gather的区分：在Scatter中，线程是分别给输入数据的，每个线程再决定输出到某个输出位置上。而Gather，线程是分配给输出数据的，每个线程可能会读取多个输入位置。

![image-20211108202128148](./images/image-20211108202128148.png)

上面代码中Gather会更加高效，因为左边Scatter的代码，会涉及到要解决不同线程之间写冲突的问题。

Gather: 会有很多重叠的读取

Scatter: 可能会有很多冲突的写入

#### Tiling

在芯片的快速存储上（比如共享内存上）缓存那些会被重复读取的数据。比如我们对图片进行模糊，则我们一般会将图像分成一块一块的Tile，然后将这一块的数据加载到SharedMemory中，然后进行模糊处理。

#### Privatization

![image-20211108203123478](./images/image-20211108203123478.png)

这个技术是指，当我们有多个线程都在写一些全局数据时，我们可以让每个线程独立拷贝一份全局数据进行私有化，这样就不会做一些线程间加锁与同步的事情。最后再用额外的线程来考虑每个线程上结果的合并。

这一技术在直方图的作业中可以使用。

#### Binning / Spatial Data Structures

Binning：Build Data structure that map output locations to the relevant input data

就是当我们不能明确我们对应位置的输入，需要哪些位置的输入时，我们可能需要把输入都遍历一遍，但这是巨大的计算浪费。所以Binning就是把input按一定的规则装到一个一个的Bin中，这个时候，我们计算输出时，只需要查看对应的一个或几个Bins就可以了。

举个例子，我们要计算每个城市可能的门店附近300Km城市的人口之和，一般做法，我们需要检查每个城市和该门店的距离，然后把小于300Km的留下进行统计计算。

![image-20211108204544857](./images/image-20211108204544857.png)

使用Binning的方法，就是对整个地图划分300Km的网络。这样我们只需要在可能的网格中的检查对应的城市就可以了。

![image-20211108204603697](./images/image-20211108204603697.png)



#### Compaction

![image-20211108205438779](./images/image-20211108205438779.png)

Compaction做的事情就是，处理的数据中，只有一部分满足计算的条件，那么就会导致wrap中有大量线程是闲置的。优化的方法，就是把那些满足条件的数据单独挑选出来，拷贝到一块新空间中，然后再用较小的线程来进行运算。

#### Regularization - load balancing

正则化是指重新组织数据来减少线程之间任务的不平衡性。

比如上面美国城市距离的例子，有一些网络里城市很少，而有一些网络中城市很多。

### Libraries

整个cuda编程或并行计算方面是个巨大的生态，在这个生态里，有NVDIA官方、第三方机构、商业组织、开源社区等都会有一些写好的，针对某一领域或通用的cuda库。

这里列举一些比较好的库，它们具有以下的特点：

1. 非常成熟稳定
1. 是经过一些优化专家设计的，具有很好的性能
1. 会随着新的GPU架构进行适配与优化的

![image-20211109192958762](./images/image-20211109192958762.png)

### Programming Power Tools

这类的库很少会针对某一具体的领域设计一些接口，而更多的是提供一些基础功能让程序员去实现自己的解决方案。我们把这一类的库称为Power Tools。

![image-20211109193626419](./images/image-20211109193626419.png)

[**CUB**](https://github.com/NVIDIA/cub): CUDA unBind, CUB provides state-of-the-art, reusable software components for every layer of the CUDA programming model.

[**CudaDMA**](http://lightsighter.github.io/CudaDMA/) is a library of DMA objects that support efficient movement of data between off-chip global memory and on-chip shared memory in CUDA kernels. CudaDMA objects support many different data transfer patterns including sequential, strided, and indirect patterns.

### Platforms

CUDA不等价于CUDA C/C++，CUDA本身是和语言无关的架构，所以我们还可以使用一些别的语言进行CUDA编程，比如Python，Matlab、Fortran等。

跨平台并行编程的框架：OpenCL、OpenGL、OpenACC，前两者和CUDA很类似，会有一些并行编程的模型，比如线程块，SharedMemory等，而OpenACC更像是OpenMP属于一种前导式编译指示符，放在一些需要并行化的语句块的前面 。

## Dynamic Parallelism

动态并行意思就是可以让kernel函数内，再启动新的kernel函数。动态并行特别适配一些嵌套并行（Nested Parallelism）和一些递归并行（Recursive Parallelism）逻辑的算法。

上边是普通的从host来调用kernel函数Hello，而下边HelloWorld本身就是一个kernel函数，它的内部再次调用了kernel函数Hello。

```cpp
__global__ void Hello() {
	printf("Hello");
}
void main() {
	Hello<<<1, 1>>>();
  cudaDeviceSynchronize();
  printf("World");
}
```

```cpp
__global__ void Hello() {
	printf("Hello");
}
__global__ void HelloWorld() {
	Hello<<<1, 1>>>();
  cudaDeviceSynchronize();
  printf("World");
}
```

动态并行编程过程中需要注意的问题：

1. 每个线程执行相同的代码段，所以在kernel函数内部的kernel函数可能会被父级的多个线程同时启动，所以一般的编程中，可能需要指定哪个线程去启动子kernel函数。

   ```cpp
   __global__ void launcher() {
     if(threadIdx.x == 0) {
       kernel<<<1,1>>>();
     }
   }
   ```
   
2. 每个线程块执行时都是彼此独立的，线程块内部的Streams和Events都是私用的。

3. 线程块内的共享存储是私有的，不可以传递给子Kernel函数。

使用动态并行编写快排的示例：

```cpp
__global__ void quicksort(int  *data, int left, int right) {
  int nleft, nright;
  cudaStream_t s1, s2;
  partiition(data+left, data+right, data[left], nleft, nright);
  
  if(left < nright) {
    cudaStreamCreateWithFlags(&s1, cudaStreamNonBlocking);
    quicksort<<<..., s1>>>(data, left, nright);
  }
  if(nleft < right) {
    cudaStreamCreateWithFlags(&s2, cudaStreamNonBlocking);
    quicksort<<<..., s2>>>(data, nleft, right);
  }
}
```
