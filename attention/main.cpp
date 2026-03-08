// main.cpp
#include <iostream>
#include <vector>
#include <chrono>
#include <cstring>
#include <iomanip>
#include <cmath>
#include <cuda_runtime.h>  // 添加这个头文件

struct AttentionTiming {
    float total_time;
    float h2d_time;
    float d2h_time;
    float alloc_time;
    float scores_kernel_time;
    float softmax_kernel_time;
    float output_kernel_time;
    float mem_peak;
};

// 声明外部C函数
extern "C" {
    AttentionTiming attention_forward_cuda(const float* h_Q, const float* h_K, const float* h_V,
                                          float* h_output, float* h_scores,
                                          int batch_size, int num_heads, int seq_len, int head_dim);
    
    void attention_forward_cpp(const float* Q, const float* K, const float* V,
                              float* output, float* scores,
                              int batch_size, int num_heads, int seq_len, int head_dim);
    
    void generate_attention_data(float* Q, float* K, float* V, int total_elements);
    bool verify_attention_results(const float* cpu_out, const float* gpu_out, 
                                  int size, float tolerance);
    
    void print_cuda_device_info();
}

int main() {
    printf("================================================\n");
    printf("Transformer注意力机制性能对比\n");
    printf("================================================\n\n");
    
    // 初始化CUDA
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        printf("错误: 没有找到CUDA设备\n");
        return 1;
    }
    
    // 打印CUDA设备信息
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("CUDA设备: %s\n", prop.name);
    printf("计算能力: %d.%d\n", prop.major, prop.minor);
    printf("全局内存: %.2f GB\n", prop.totalGlobalMem / 1024.0 / 1024.0 / 1024.0);
    printf("================================================\n\n");
    
    // 测试不同规模的注意力计算
    struct AttentionConfig {
        int batch_size;
        int num_heads;
        int seq_len;
        int head_dim;
        const char* name;
        float expected_gflops;
    };
    
    std::vector<AttentionConfig> configs = {
        {1, 1, 64, 64, "tiny", 0.05},      // 小模型
        {1, 8, 128, 64, "small", 0.8},     // 小BERT
        {2, 12, 256, 64, "medium", 6.4},   // BERT-base
        {4, 16, 512, 64, "large", 51.2},   // BERT-large
        // {8, 16, 1024, 64, "xlarge", 409.6} // 如果内存允许可以启用
    };
    
    printf("\n=== Transformer注意力机制 CUDA vs CPU 性能对比 ===\n\n");
    
    // 表格头部
    printf("%-15s %-12s %-12s %-12s %-12s %-10s %-12s\n",
           "配置", "CPU(ms)", "GPU核(ms)", "GPU总(ms)", "加速比", "验证", "GFLOPS");
    printf("%-15s %-12s %-12s %-12s %-12s %-10s %-12s\n",
           "---------------", "-----------", "-----------", "-----------", "-----------", "--------", "-----------");
    
    for (const auto& cfg : configs) {
        int batch = cfg.batch_size;
        int heads = cfg.num_heads;
        int seq = cfg.seq_len;
        int dim = cfg.head_dim;
        
        int total_heads = batch * heads;
        int total_elements = total_heads * seq * dim;
        int scores_elements = total_heads * seq * seq;
        
        // 检查内存是否足够
        size_t total_memory_needed = (3 * total_elements + scores_elements + total_elements) * sizeof(float);
        size_t free_mem, total_mem;
        cudaMemGetInfo(&free_mem, &total_mem);  // 现在可以正常使用了
        
        if (total_memory_needed > free_mem) {
            printf("\n配置: %s 需要 %.2f MB GPU内存，但只有 %.2f MB 可用，跳过\n",
                   cfg.name, total_memory_needed / 1024.0 / 1024.0, free_mem / 1024.0 / 1024.0);
            continue;
        }
        
        printf("\n处理配置: %s (batch=%d, heads=%d, seq=%d, dim=%d)\n",
               cfg.name, batch, heads, seq, dim);
        printf("数据大小:\n");
        printf("  Q/K/V: %.2f MB each\n", total_elements * sizeof(float) / 1024.0 / 1024.0);
        printf("  Scores: %.2f MB\n", scores_elements * sizeof(float) / 1024.0 / 1024.0);
        printf("  Output: %.2f MB\n", total_elements * sizeof(float) / 1024.0 / 1024.0);
        printf("  总GPU内存需求: %.2f MB\n", total_memory_needed / 1024.0 / 1024.0);
        printf("  可用GPU内存: %.2f MB\n", free_mem / 1024.0 / 1024.0);
        printf("--------------------------------------------------------\n");
        
        // 分配内存
        float* h_Q = new float[total_elements];
        float* h_K = new float[total_elements];
        float* h_V = new float[total_elements];
        float* h_output_cpu = new float[total_elements];
        float* h_output_gpu = new float[total_elements];
        float* h_scores_cpu = new float[scores_elements];
        float* h_scores_gpu = new float[scores_elements];
        
        // 生成测试数据
        printf("生成测试数据...\n");
        generate_attention_data(h_Q, h_K, h_V, total_elements);
        
        // CPU版本
        printf("\n运行CPU版本...\n");
        auto cpu_start = std::chrono::high_resolution_clock::now();
        attention_forward_cpp(h_Q, h_K, h_V, h_output_cpu, h_scores_cpu,
                             batch, heads, seq, dim);
        auto cpu_end = std::chrono::high_resolution_clock::now();
        double cpu_time = std::chrono::duration<double, std::milli>(cpu_end - cpu_start).count();
        printf("  CPU执行时间: %.3f ms\n", cpu_time);
        
        // GPU版本
        printf("\n运行GPU版本...\n");
        AttentionTiming timing = attention_forward_cuda(h_Q, h_K, h_V, 
                                                        h_output_gpu, h_scores_gpu,
                                                        batch, heads, seq, dim);
        
        printf("\n  GPU性能详细分析:\n");
        printf("  ----------------------------------------\n");
        printf("  内存分配/释放:     %.3f ms\n", timing.alloc_time);
        printf("  数据传输(H2D):     %.3f ms (%.2f MB)\n", 
               timing.h2d_time, 3 * total_elements * sizeof(float) / 1024.0 / 1024.0);
        printf("  数据传输(D2H):     %.3f ms (%.2f MB)\n", 
               timing.d2h_time, (total_elements + scores_elements) * sizeof(float) / 1024.0 / 1024.0);
        printf("  核函数 - Scores:   %.3f ms\n", timing.scores_kernel_time);
        printf("  核函数 - Softmax:  %.3f ms\n", timing.softmax_kernel_time);
        printf("  核函数 - Output:   %.3f ms\n", timing.output_kernel_time);
        printf("  核函数总时间:      %.3f ms\n", 
               timing.scores_kernel_time + timing.softmax_kernel_time + timing.output_kernel_time);
        printf("  ----------------------------------------\n");
        printf("  峰值GPU内存:       %.2f MB\n", timing.mem_peak);
        printf("  GPU总时间:         %.3f ms\n", timing.total_time);
        printf("  ========================================\n");
        
        // 验证结果
        printf("\n验证结果: ");
        bool verified = verify_attention_results(h_output_cpu, h_output_gpu, 
                                                total_elements, 1e-4f);
        
        // 计算GFLOPS
        double flops_scores = 2.0 * total_heads * seq * seq * dim;  // Q*K^T
        double flops_softmax = 3.0 * total_heads * seq * seq;       // softmax
        double flops_output = 2.0 * total_heads * seq * seq * dim;  // softmax*V
        double total_flops = (flops_scores + flops_softmax + flops_output) / 1e9; // GFLOPS
        double gpu_gflops = total_flops / (timing.scores_kernel_time + 
                                          timing.softmax_kernel_time + 
                                          timing.output_kernel_time) * 1000;
        
        // 输出表格行
        char config_name[15];
        sprintf(config_name, "%s", cfg.name);
        printf("\n%-15s %-12.3f %-12.3f %-12.3f %-12.2f %-10s %-12.2f\n",
               config_name,
               cpu_time,
               timing.scores_kernel_time + timing.softmax_kernel_time + timing.output_kernel_time,
               timing.total_time,
               cpu_time / timing.total_time,
               verified ? "✓" : "✗",
               gpu_gflops);
        
        printf("========================================================\n");
        
        // 释放内存
        delete[] h_Q;
        delete[] h_K;
        delete[] h_V;
        delete[] h_output_cpu;
        delete[] h_output_gpu;
        delete[] h_scores_cpu;
        delete[] h_scores_gpu;
    }
    
    printf("\n\n性能总结:\n");
    printf("================================================\n");
    printf("加速比说明:\n");
    printf("  - GPU核时间: 三个核函数执行时间总和\n");
    printf("  - GPU总时间: 包括内存分配、数据传输和核函数执行\n");
    printf("  - 加速比: CPU时间 / GPU总时间\n");
    printf("  - GFLOPS: GPU实际计算性能\n");
    printf("  - 注意力机制复杂度O(n²)，序列越长加速比越高\n");
    printf("================================================\n");
    
    // 重置CUDA设备
    cudaDeviceReset();
    
    return 0;
}
