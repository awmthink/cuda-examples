# Examples: Hello

本示例程序演示了在Host侧进行简单的的kenrel函数的调用，在Device侧，由kernel函数打印"hello"字符串，然后在Host侧打印"world”字符串。

本示例中调用kernel函数时，指定了`<<<1,1>>>`，说明仅仅只启动了1个线程，从打印结果上也可以看出，只有一个Device侧线程打印了"hello"。

在kernel函数调用后，调用了`cudaDeviceSynchronize()`，该函数用于等待Device侧的kernel函数执行完成。如果不加该函数调用，则Host侧代码在打印"world"后，会直接退出。

在kernel函数中调用`printf`进行打印在早期的CUDA版本中是不支持的，从Compute Capability 2.0后开始支持。