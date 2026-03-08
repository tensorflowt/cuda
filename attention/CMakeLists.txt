cmake_minimum_required(VERSION 3.18)
project(attention_demo LANGUAGES CXX CUDA)

# 设置C++标准
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# 设置输出目录
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# 查找CUDA
find_package(CUDA REQUIRED)

# 包含CUDA头文件目录
include_directories(${CUDA_INCLUDE_DIRS})

# 添加可执行文件
add_executable(attention_demo
    main.cpp
    attention_cpp.cpp
    attention_cuda.cu
)

# 设置CUDA架构
set_target_properties(attention_demo PROPERTIES
    CUDA_ARCHITECTURES "75"  # GTX 1650 Ti
)

# 链接CUDA库
target_link_libraries(attention_demo
    PRIVATE
    ${CUDA_LIBRARIES}
    cudart  # 显式链接cudart
)

# 显示CUDA信息
message(STATUS "CUDA Version: ${CUDA_VERSION}")
message(STATUS "CUDA Include: ${CUDA_INCLUDE_DIRS}")
message(STATUS "CUDA Libraries: ${CUDA_LIBRARIES}")
