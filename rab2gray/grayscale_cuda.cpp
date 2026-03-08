#include <stdio.h>
#include <cuda_runtime.h>

// CUDA错误检查宏
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            printf("CUDA Error at %s:%d - %s\n", __FILE__, __LINE__, \
                   cudaGetErrorString(err)); \
            exit(1); \
        } \
    } while(0)

// 定义时间结构
struct GpuTiming {
    float total_time;
    float kernel_time;
    float transfer_time;
    float alloc_time;
};

// CUDA核函数
__global__ void grayscale_kernel(const unsigned char* rgb, 
                                 unsigned char* gray, 
                                 int width, 
                                 int height) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    
    if (x < width && y < height) {
        
        int rgb_idx = (y * width + x) * 3;
        int gray_idx = y * width + x;
        
        unsigned char r = rgb[rgb_idx];
        unsigned char g = rgb[rgb_idx + 1];
        unsigned char b = rgb[rgb_idx + 2];
        
        // 整数运算: 0.299*R + 0.587*G + 0.114*B
        gray[gray_idx] = (299 * r + 587 * g + 114 * b) / 1000;
    }
}

// CUDA版本的封装函数 - 返回详细的时间信息
extern "C" GpuTiming convert_to_gray_cuda(const unsigned char* h_rgb, unsigned char* h_gray, int width, int height) {
    unsigned char *d_rgb, *d_gray;
    size_t rgb_size = width * height * 3 * sizeof(unsigned char);
    size_t gray_size = width * height * sizeof(unsigned char);
    
    GpuTiming timing = {0, 0, 0, 0};
    
    // 创建CUDA事件
    cudaEvent_t total_start, total_stop;
    cudaEvent_t alloc_start, alloc_stop;
    cudaEvent_t transfer_start, transfer_stop;
    cudaEvent_t kernel_start, kernel_stop;
    
    cudaEventCreate(&total_start);
    cudaEventCreate(&total_stop);
    cudaEventCreate(&alloc_start);
    cudaEventCreate(&alloc_stop);
    cudaEventCreate(&transfer_start);
    cudaEventCreate(&transfer_stop);
    cudaEventCreate(&kernel_start);
    cudaEventCreate(&kernel_stop);
    
    // 开始总计时
    cudaEventRecord(total_start);
    
    // ===== 内存分配阶段 =====
    cudaEventRecord(alloc_start);
    
    cudaMalloc(&d_rgb, rgb_size);
    cudaMalloc(&d_gray, gray_size);
    
    cudaEventRecord(alloc_stop);
    
    // ===== 数据传输到GPU (H2D) =====
    cudaEventRecord(transfer_start);
    
    cudaMemcpy(d_rgb, h_rgb, rgb_size, cudaMemcpyHostToDevice);
    
    cudaEventRecord(transfer_stop);
    
    // ===== 核函数执行 =====
    // 设置CUDA核函数参数
    dim3 block(16, 16); // 256线程，建议block大小为32倍数，warp大小
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y); // ceil除法（向上取整）
    
    cudaEventRecord(kernel_start);
    
    // 调用核函数
    grayscale_kernel<<<grid, block>>>(d_rgb, d_gray, width, height);
    
    // 等待核函数完成
    cudaDeviceSynchronize();
    
    cudaEventRecord(kernel_stop);
    
    // ===== 数据传输回CPU (D2H) =====
    cudaEventRecord(transfer_start);
    
    cudaMemcpy(h_gray, d_gray, gray_size, cudaMemcpyDeviceToHost);
    
    cudaEventRecord(transfer_stop);
    
    // ===== 内存释放 =====
    cudaEventRecord(alloc_start);
    
    cudaFree(d_rgb);
    cudaFree(d_gray);
    
    cudaEventRecord(alloc_stop);
    
    // 结束总计时
    cudaEventRecord(total_stop);
    cudaEventSynchronize(total_stop);
    
    // 计算各个阶段的时间
    float alloc_time = 0, transfer_time = 0, kernel_time = 0, total_time = 0;
    
    cudaEventElapsedTime(&total_time, total_start, total_stop);
    cudaEventElapsedTime(&kernel_time, kernel_start, kernel_stop);
    cudaEventElapsedTime(&transfer_time, transfer_start, transfer_stop);
    cudaEventElapsedTime(&alloc_time, alloc_start, alloc_stop);
    
    // 填充时间结构
    timing.total_time = total_time;
    timing.kernel_time = kernel_time;
    timing.transfer_time = transfer_time;
    timing.alloc_time = alloc_time;
    
    printf("  GPU性能详细分析:\n");
    printf("  ----------------------------------------\n");
    printf("  内存分配/释放:     %.3f ms\n", alloc_time);
    printf("  数据传输(H2D+D2H): %.3f ms\n", transfer_time);
    printf("  其中: H2D (%.2f MB), D2H (%.2f MB)\n", 
           rgb_size/1024.0/1024.0, gray_size/1024.0/1024.0);
    printf("  核函数执行:        %.3f ms\n", kernel_time);
    printf("  ----------------------------------------\n");
    printf("  GPU总时间:         %.3f ms\n", total_time);
    printf("  ========================================\n");
    
    // 释放事件
    cudaEventDestroy(total_start);
    cudaEventDestroy(total_stop);
    cudaEventDestroy(alloc_start);
    cudaEventDestroy(alloc_stop);
    cudaEventDestroy(transfer_start);
    cudaEventDestroy(transfer_stop);
    cudaEventDestroy(kernel_start);
    cudaEventDestroy(kernel_stop);
    
    return timing;
}
