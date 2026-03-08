// main.cpp
#include <iostream>
#include <vector>
#include <chrono>
#include <cuda_runtime.h>
#include <cstring>
#include <iomanip>

// 修改函数声明，让CUDA函数返回更多信息
struct GpuTiming {
    float total_time;
    float kernel_time;
    float transfer_time;
    float alloc_time;
};

// 声明外部C函数
extern "C" {
    GpuTiming convert_to_gray_cuda(const unsigned char* h_rgb, unsigned char* h_gray, int width, int height);
    void convert_to_gray_cpp(const unsigned char* rgb, unsigned char* gray, int width, int height);
    void generate_test_data(unsigned char* rgb, int width, int height);
    bool verify_results(const unsigned char* a, const unsigned char* b, int size, float tolerance);
}

int main() {
    // 检查CUDA设备
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        printf("错误: 没有找到CUDA设备\n");
        return 1;
    }
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("================================================\n");
    printf("CUDA设备: %s\n", prop.name);
    printf("计算能力: %d.%d\n", prop.major, prop.minor);
    printf("================================================\n\n");
    
    // 测试不同大小的图像
    std::vector<std::pair<int, int>> test_sizes = {
        {512, 512},      // 0.25 MP
        {1024, 1024},    // 1 MP
        {1920, 1080},    // 2 MP (Full HD)
        {3840, 2160},    // 8 MP (4K)
        {7680, 4320}     // 33 MP (8K)
    };
    
    printf("=== CUDA vs CPU 灰度转换性能对比 ===\n\n");
    
    // 创建表格头部
    printf("%-15s %-12s %-15s %-15s %-15s %-10s\n", 
           "图像尺寸", "CPU(ms)", "GPU核函数(ms)", "GPU总时间(ms)", "加速比", "验证");
    printf("%-15s %-12s %-15s %-15s %-15s %-10s\n",
           "-------------", "----------", "---------------", "---------------", "----------", "--------");
    
    for (const auto& size : test_sizes) {
        int width = size.first;
        int height = size.second;
        size_t rgb_size = width * height * 3;
        size_t gray_size = width * height;
        
        printf("\n处理图像: %dx%d (%.2f MP)\n", width, height, width * height / 1000000.0);
        printf("数据大小: %.2f MB RGB, %.2f MB 灰度\n", 
               rgb_size / 1024.0 / 1024.0, gray_size / 1024.0 / 1024.0);
        printf("--------------------------------------------------------\n");
        
        // 分配主机内存
        unsigned char* h_rgb = new unsigned char[rgb_size];
        unsigned char* h_gray_cpu = new unsigned char[gray_size];
        unsigned char* h_gray_gpu = new unsigned char[gray_size];
        
        // 生成测试数据
        printf("生成测试数据...\n");
        generate_test_data(h_rgb, width, height);
        
        // 运行测试
        double cpu_time = 0;
        GpuTiming gpu_timing;
        
        // CPU版本
        printf("\n运行CPU版本...\n");
        auto cpu_start = std::chrono::high_resolution_clock::now();
        convert_to_gray_cpp(h_rgb, h_gray_cpu, width, height);
        auto cpu_end = std::chrono::high_resolution_clock::now();
        cpu_time = std::chrono::duration<double, std::milli>(cpu_end - cpu_start).count();
        printf("  CPU执行时间: %.3f ms\n", cpu_time);
        
        // GPU版本
        printf("\n运行GPU版本...\n");
        gpu_timing = convert_to_gray_cuda(h_rgb, h_gray_gpu, width, height);
        
        // 验证结果
        printf("\n验证结果: ");
        bool verified = verify_results(h_gray_cpu, h_gray_gpu, gray_size, 1.0f);
        
        // 输出表格行 - 使用实际的核函数时间，而不是估算值
        char size_str[20];
        sprintf(size_str, "%dx%d", width, height);
        
        printf("\n%-15s %-12.2f %-15.3f %-15.3f %-15.2f %-10s\n", 
               size_str,
               cpu_time,
               gpu_timing.kernel_time,  // 使用实际的核函数时间
               gpu_timing.total_time,
               cpu_time / gpu_timing.total_time,
               verified ? "✓" : "✗");
        
        // 显示详细的时间分解
        printf("  详细: 核函数=%.3fms, 传输=%.3fms, 分配=%.3fms\n",
               gpu_timing.kernel_time,
               gpu_timing.transfer_time,
               gpu_timing.alloc_time);
        
        printf("========================================================\n");
        
        // 释放内存
        delete[] h_rgb;
        delete[] h_gray_cpu;
        delete[] h_gray_gpu;
    }
    
    printf("\n\n性能总结:\n");
    printf("================================================\n");
    printf("加速比说明:\n");
    printf("  - CPU时间: 纯CPU计算时间\n");
    printf("  - GPU核函数时间: 纯GPU计算时间（实际测量值）\n");
    printf("  - GPU总时间: 包括内存分配、数据传输和核函数执行\n");
    printf("  - 加速比: CPU时间 / GPU总时间\n");
    printf("================================================\n");
    
    // 检查CUDA错误并重置设备
    cudaDeviceReset();
    
    return 0;
}
