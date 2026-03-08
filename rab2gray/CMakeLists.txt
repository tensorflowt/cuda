cmake_minimum_required(VERSION 3.10)
project(GrayscaleComparison)

set(CMAKE_CXX_STANDARD 11)

# 查找CUDA
find_package(CUDA REQUIRED)

# 添加CUDA源文件
cuda_add_executable(grayscale_comparison 
    main.cpp
    grayscale_cpp.cpp
    grayscale_cuda.cu
)

# 设置编译选项
target_compile_features(grayscale_comparison PRIVATE cxx_std_11)

# 设置输出目录
set_target_properties(grayscale_comparison PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)
