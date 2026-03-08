// attention_cuda.cu
#include <stdio.h>
#include <cuda_runtime.h>
#include <math.h>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            printf("CUDA Error at %s:%d - %s\n", __FILE__, __LINE__, \
                   cudaGetErrorString(err)); \
            exit(1); \
        } \
    } while(0)

// 时间结构
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

// Q * K^T 核函数
__global__ void scores_kernel(const float* Q, const float* K, float* scores,
                             int total_heads, int seq_len, int head_dim) {
    int b = blockIdx.z;      // batch head index
    int i = blockIdx.x * blockDim.x + threadIdx.x;  // query position
    int j = blockIdx.y * blockDim.y + threadIdx.y;  // key position
    
    if (i < seq_len && j < seq_len) {
        float sum = 0.0f;
        int base_idx = b * seq_len * head_dim;
        
        // 向量点积
        for (int d = 0; d < head_dim; d++) {
            sum += Q[base_idx + i * head_dim + d] * K[base_idx + j * head_dim + d];
        }
        
        // 缩放
        sum /= sqrtf((float)head_dim);
        
        // 存储分数
        scores[b * seq_len * seq_len + i * seq_len + j] = sum;
    }
}

// Softmax核函数 - 修正版 (支持任意序列长度)
__global__ void softmax_kernel(float* scores, int total_heads, int seq_len) {
    int b = blockIdx.x;  // batch head index
    int tid = threadIdx.x;
    
    // 每个线程处理多行，确保所有行都被处理
    for (int row = tid; row < seq_len; row += blockDim.x) {
        int row_start = b * seq_len * seq_len + row * seq_len;
        
        // 找最大值（数值稳定性）
        float max_val = scores[row_start];
        for (int j = 1; j < seq_len; j++) {
            max_val = fmaxf(max_val, scores[row_start + j]);
        }
        
        // 计算exp和
        float sum = 0.0f;
        for (int j = 0; j < seq_len; j++) {
            sum += expf(scores[row_start + j] - max_val);
        }
        
        // 应用softmax
        for (int j = 0; j < seq_len; j++) {
            scores[row_start + j] = expf(scores[row_start + j] - max_val) / sum;
        }
    }
}

// 输出核函数 (softmax * V)
__global__ void output_kernel(const float* scores, const float* V, float* output,
                             int total_heads, int seq_len, int head_dim) {
    int b = blockIdx.z;      // batch head index
    int i = blockIdx.y * blockDim.y + threadIdx.y;  // query position
    int d = blockIdx.x * blockDim.x + threadIdx.x;  // dimension
    
    if (i < seq_len && d < head_dim) {
        float sum = 0.0f;
        int base_idx = b * seq_len * head_dim;
        int score_base = b * seq_len * seq_len;
        
        // 注意力加权求和
        for (int j = 0; j < seq_len; j++) {
            float weight = scores[score_base + i * seq_len + j];
            sum += weight * V[base_idx + j * head_dim + d];
        }
        
        output[base_idx + i * head_dim + d] = sum;
    }
}

// 获取最优block大小的函数
int get_optimal_block_size(int seq_len) {
    int block_size = 256;  // 默认值
    
    // 确保block_size >= seq_len
    while (block_size < seq_len) {
        block_size *= 2;
        if (block_size > 1024) {  // 硬件限制
            block_size = 1024;
            break;
        }
    }
    
    return block_size;
}

// 完整的注意力前向传播带详细日志
extern "C" AttentionTiming attention_forward_cuda(const float* h_Q, const float* h_K, const float* h_V,
                                                  float* h_output, float* h_scores,
                                                  int batch_size, int num_heads, int seq_len, int head_dim) {
    
    AttentionTiming timing = {0};
    int total_heads = batch_size * num_heads;
    size_t qkv_size = total_heads * seq_len * head_dim * sizeof(float);
    size_t scores_size = total_heads * seq_len * seq_len * sizeof(float);
    size_t output_size = total_heads * seq_len * head_dim * sizeof(float);
    
    float *d_Q, *d_K, *d_V, *d_scores, *d_output;
    
    // 创建CUDA事件
    cudaEvent_t total_start, total_stop;
    cudaEvent_t alloc_start, alloc_stop;
    cudaEvent_t h2d_start, h2d_stop;
    cudaEvent_t d2h_start, d2h_stop;
    cudaEvent_t scores_start, scores_stop;
    cudaEvent_t softmax_start, softmax_stop;
    cudaEvent_t output_start, output_stop;
    
    cudaEventCreate(&total_start);
    cudaEventCreate(&total_stop);
    cudaEventCreate(&alloc_start);
    cudaEventCreate(&alloc_stop);
    cudaEventCreate(&h2d_start);
    cudaEventCreate(&h2d_stop);
    cudaEventCreate(&d2h_start);
    cudaEventCreate(&d2h_stop);
    cudaEventCreate(&scores_start);
    cudaEventCreate(&scores_stop);
    cudaEventCreate(&softmax_start);
    cudaEventCreate(&softmax_stop);
    cudaEventCreate(&output_start);
    cudaEventCreate(&output_stop);
    
    cudaEventRecord(total_start);
    
    // ===== 内存分配 =====
    cudaEventRecord(alloc_start);
    printf("    分配GPU内存...\n");
    CUDA_CHECK(cudaMalloc(&d_Q, qkv_size));
    CUDA_CHECK(cudaMalloc(&d_K, qkv_size));
    CUDA_CHECK(cudaMalloc(&d_V, qkv_size));
    CUDA_CHECK(cudaMalloc(&d_scores, scores_size));
    CUDA_CHECK(cudaMalloc(&d_output, output_size));
    cudaEventRecord(alloc_stop);
    
    // ===== 数据传输到GPU =====
    cudaEventRecord(h2d_start);
    printf("    复制数据到GPU (Q/K/V)...\n");
    CUDA_CHECK(cudaMemcpy(d_Q, h_Q, qkv_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_K, h_K, qkv_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_V, h_V, qkv_size, cudaMemcpyHostToDevice));
    cudaEventRecord(h2d_stop);
    
    // ===== 核函数执行 =====
    
    // 1. 注意力分数核函数配置
    dim3 block1(16, 16);
    dim3 grid1(
        (seq_len + block1.x - 1) / block1.x,
        (seq_len + block1.y - 1) / block1.y,
        total_heads
    );
    
    // 2. Softmax核函数配置 - 动态调整block大小
    int softmax_block_size = get_optimal_block_size(seq_len);
    dim3 block2(softmax_block_size);
    dim3 grid2(total_heads);  // 每个head一个block
    
    // 3. 输出核函数配置
    dim3 block3(16, 16);
    dim3 grid3(
        (head_dim + block3.x - 1) / block3.x,
        (seq_len + block3.y - 1) / block3.y,
        total_heads
    );
    
    // 执行scores核函数
    cudaEventRecord(scores_start);
    printf("    启动scores核函数: grid(%d,%d,%d), block(%d,%d)\n", 
           grid1.x, grid1.y, grid1.z, block1.x, block1.y);
    scores_kernel<<<grid1, block1>>>(d_Q, d_K, d_scores, total_heads, seq_len, head_dim);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    cudaEventRecord(scores_stop);
    
    // 执行softmax核函数
    cudaEventRecord(softmax_start);
    printf("    启动softmax核函数: grid(%d), block(%d) (每个线程处理多行, seq_len=%d)\n", 
           grid2.x, block2.x, seq_len);
    softmax_kernel<<<grid2, block2>>>(d_scores, total_heads, seq_len);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    cudaEventRecord(softmax_stop);
    
    // 执行output核函数
    cudaEventRecord(output_start);
    printf("    启动output核函数: grid(%d,%d,%d), block(%d,%d)\n", 
           grid3.x, grid3.y, grid3.z, block3.x, block3.y);
    output_kernel<<<grid3, block3>>>(d_scores, d_V, d_output, total_heads, seq_len, head_dim);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    cudaEventRecord(output_stop);
    
    // ===== 数据传输回CPU =====
    cudaEventRecord(d2h_start);
    printf("    复制结果回CPU (output+scores)...\n");
    CUDA_CHECK(cudaMemcpy(h_output, d_output, output_size, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_scores, d_scores, scores_size, cudaMemcpyDeviceToHost));
    cudaEventRecord(d2h_stop);
    
    // ===== 内存释放 =====
    cudaEventRecord(alloc_start);
    CUDA_CHECK(cudaFree(d_Q));
    CUDA_CHECK(cudaFree(d_K));
    CUDA_CHECK(cudaFree(d_V));
    CUDA_CHECK(cudaFree(d_scores));
    CUDA_CHECK(cudaFree(d_output));
    cudaEventRecord(alloc_stop);
    
    cudaEventRecord(total_stop);
    cudaEventSynchronize(total_stop);
    
    // 计算时间
    cudaEventElapsedTime(&timing.alloc_time, alloc_start, alloc_stop);
    cudaEventElapsedTime(&timing.h2d_time, h2d_start, h2d_stop);
    cudaEventElapsedTime(&timing.d2h_time, d2h_start, d2h_stop);
    cudaEventElapsedTime(&timing.scores_kernel_time, scores_start, scores_stop);
    cudaEventElapsedTime(&timing.softmax_kernel_time, softmax_start, softmax_stop);
    cudaEventElapsedTime(&timing.output_kernel_time, output_start, output_stop);
    cudaEventElapsedTime(&timing.total_time, total_start, total_stop);
    
    // 计算峰值内存 (Q, K, V, scores, output 同时在GPU上)
    timing.mem_peak = (3 * qkv_size + scores_size + output_size) / 1024.0 / 1024.0;
    
    // 销毁事件
    cudaEventDestroy(total_start);
    cudaEventDestroy(total_stop);
    cudaEventDestroy(alloc_start);
    cudaEventDestroy(alloc_stop);
    cudaEventDestroy(h2d_start);
    cudaEventDestroy(h2d_stop);
    cudaEventDestroy(d2h_start);
    cudaEventDestroy(d2h_stop);
    cudaEventDestroy(scores_start);
    cudaEventDestroy(scores_stop);
    cudaEventDestroy(softmax_start);
    cudaEventDestroy(softmax_stop);
    cudaEventDestroy(output_start);
    cudaEventDestroy(output_stop);
    
    return timing;
}

// 获取CUDA设备信息的函数
extern "C" void print_cuda_device_info() {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    
    printf("================================================\n");
    printf("找到 %d 个CUDA设备:\n", deviceCount);
    
    for (int i = 0; i < deviceCount; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        printf("设备 %d: %s\n", i, prop.name);
        printf("  计算能力: %d.%d\n", prop.major, prop.minor);
        printf("  多处理器数量: %d\n", prop.multiProcessorCount);
        printf("  全局内存: %.2f GB\n", prop.totalGlobalMem / 1024.0 / 1024.0 / 1024.0);
        printf("  内存带宽: %.2f GB/s\n", 
               2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1e6);
        printf("  最大线程数/块: %d\n", prop.maxThreadsPerBlock);
        printf("  最大块维度: %d x %d x %d\n", 
               prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
        printf("  最大网格维度: %d x %d x %d\n", 
               prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
        printf("  SM数量: %d\n", prop.multiProcessorCount);
        printf("  warp大小: %d\n", prop.warpSize);
    }
    printf("================================================\n");
}
