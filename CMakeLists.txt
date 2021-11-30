cmake_minimum_required(VERSION 3.16)
project(cuda_exampels)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

enable_language(CUDA)
find_package(CUDAToolkit REQUIRED)
set(CMAKE_CUDA_ARCHITECTURES "61")

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(examples 
    examples/hello/hello.cu
    examples/device/device_query.cpp
    examples/reduce/reduce_sum.cu
)

foreach(example_src ${examples})
    get_filename_component(example_name ${example_src} NAME_WE)
    add_executable(${example_name} ${example_src})
    target_include_directories(${example_name} PRIVATE 
        ${CMAKE_SOURCE_DIR}/examples 
        ${CUDAToolkit_INCLUDE_DIRS}
    )
    target_link_libraries(${example_name} PRIVATE CUDA::cudart CUDA::cublas)
endforeach()

message(STATUS ${CMAKE_SOURCE_DIR}/examples )