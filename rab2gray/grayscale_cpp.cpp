#include <iostream>
#include <chrono>
#include <random>

extern "C" {

// C++版本的灰度转换函数
// RGB数据存放排列位置：RGB RGB RGB RGB ... 
void convert_to_gray_cpp(const unsigned char* rgb, unsigned char* gray, int width, int height) {
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int rgb_idx = (y * width + x) * 3;
            int gray_idx = y * width + x;
            
            unsigned char r = rgb[rgb_idx];
            unsigned char g = rgb[rgb_idx + 1];
            unsigned char b = rgb[rgb_idx + 2];
            
            gray[gray_idx] = (299 * r + 587 * g + 114 * b) / 1000;
        }
    }
}

// 生成测试数据
void generate_test_data(unsigned char* rgb, int width, int height) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, 255);
    
    for (int i = 0; i < width * height * 3; i++) {
        rgb[i] = dis(gen);
    }
}

// 验证结果
bool verify_results(const unsigned char* a, const unsigned char* b, int size, float tolerance) {
    int errors = 0;
    const int max_errors = 10;
    
    for (int i = 0; i < size; i++) {
        int diff = abs(static_cast<int>(a[i]) - static_cast<int>(b[i]));
        if (diff > tolerance) {
            if (errors < max_errors) {
                printf("    错误位置 %d: CPU=%d, GPU=%d, 差异=%d\n", 
                       i, a[i], b[i], diff);
            }
            errors++;
        }
    }
    
    if (errors == 0) {
        printf("✓ 结果完全一致\n");
        return true;
    } else {
        printf("✗ 发现 %d 个错误 (显示前%d个)\n", errors, max_errors);
        return false;
    }
}

} // extern "C"
