// attention_cpp.cpp
#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>

extern "C" {

// C++版本的注意力前向传播
void attention_forward_cpp(const float* Q, const float* K, const float* V,
                          float* output, float* scores,
                          int batch_size, int num_heads, int seq_len, int head_dim) {
    
    int total_heads = batch_size * num_heads;
    
    for (int h = 0; h < total_heads; h++) {
        for (int i = 0; i < seq_len; i++) {
            // 计算注意力分数 Q_i * K_j^T
            for (int j = 0; j < seq_len; j++) {
                float sum = 0.0f;
                for (int d = 0; d < head_dim; d++) {
                    int q_idx = h * seq_len * head_dim + i * head_dim + d;
                    int k_idx = h * seq_len * head_dim + j * head_dim + d;
                    sum += Q[q_idx] * K[k_idx];
                }
                sum /= sqrtf((float)head_dim);
                scores[h * seq_len * seq_len + i * seq_len + j] = sum;
            }
            
            // Softmax
            float max_val = -INFINITY;
            int row_start = h * seq_len * seq_len + i * seq_len;
            
            // 找最大值
            for (int j = 0; j < seq_len; j++) {
                max_val = fmaxf(max_val, scores[row_start + j]);
            }
            
            // 计算exp和
            float sum_exp = 0.0f;
            for (int j = 0; j < seq_len; j++) {
                sum_exp += expf(scores[row_start + j] - max_val);
            }
            
            // 应用softmax
            for (int j = 0; j < seq_len; j++) {
                scores[row_start + j] = expf(scores[row_start + j] - max_val) / sum_exp;
            }
        }
    }
    
    // 计算最终输出
    for (int h = 0; h < total_heads; h++) {
        for (int i = 0; i < seq_len; i++) {
            for (int d = 0; d < head_dim; d++) {
                float sum = 0.0f;
                for (int j = 0; j < seq_len; j++) {
                    float weight = scores[h * seq_len * seq_len + i * seq_len + j];
                    int v_idx = h * seq_len * head_dim + j * head_dim + d;
                    sum += weight * V[v_idx];
                }
                output[h * seq_len * head_dim + i * head_dim + d] = sum;
            }
        }
    }
}

// 生成测试数据
void generate_attention_data(float* Q, float* K, float* V, int total_elements) {
    for (int i = 0; i < total_elements; i++) {
        Q[i] = (float)(rand() % 1000) / 1000.0f;
        K[i] = (float)(rand() % 1000) / 1000.0f;
        V[i] = (float)(rand() % 1000) / 1000.0f;
    }
}

// 验证结果
bool verify_attention_results(const float* cpu_out, const float* gpu_out, 
                              int size, float tolerance) {
    int errors = 0;
    const int max_errors = 10;
    float max_diff = 0.0f;
    
    for (int i = 0; i < size; i++) {
        float diff = fabs(cpu_out[i] - gpu_out[i]);
        max_diff = fmaxf(max_diff, diff);
        if (diff > tolerance) {
            if (errors < max_errors) {
                printf("    位置 %d: CPU=%f, GPU=%f, 差异=%f\n", 
                       i, cpu_out[i], gpu_out[i], diff);
            }
            errors++;
        }
    }
    
    printf("  最大差异: %f\n", max_diff);
    
    if (errors == 0) {
        printf("  ✓ 结果完全一致\n");
        return true;
    } else {
        printf("  ✗ 发现 %d 个错误 (显示前%d个)\n", errors, max_errors);
        return false;
    }
}

} // extern "C"
