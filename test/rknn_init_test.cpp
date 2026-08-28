#include <iostream>
#include "rknn_api.h"
#include <string>
#include <fstream>
#include <iostream>

// 传入路径，传出大小，返回动态分配的内存指针
unsigned char* load_model_file(const std::string& path, int* size) {
    // ios::ate 表示打开时游标直接定位到文件末尾，方便查大小
    std::ifstream file(path, std::ios::in | std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        std::cerr << "Open file failed: " << path << std::endl;
        return nullptr;
    }

    *size = file.tellg(); // 获取当前游标位置（注意是游标位置, 属于相对位置即文件总字节数）
    file.seekg(0, std::ios::beg); // 把游标移回文件开头准备读取, 0表示偏移量, std::ios::beg就是begin的缩写, 表示基准位置,意思就是开头偏移0个字节,

    unsigned char* model_data = new unsigned char[*size];
    file.read(reinterpret_cast<char*>(model_data), *size);//reinterpret_cast是强制二进制类型转换，**只改编译器看待这块内存的类型，不改动数据本身**
    file.close();

    return model_data;
}

int main() {
    std::string model_path = "./models/rknn/yolo11s_640.rknn";
    int model_size = 0;

    // 1. C++ 侧加载二进制
    unsigned char* model_data = load_model_file(model_path, &model_size);
    if (!model_data) return -1;
    std::cout << "Read model bytes: " << model_size << std::endl;

    // 2. RKNN 初始化
    rknn_context ctx = 0;
    int ret = rknn_init(&ctx, model_data, model_size, 0, nullptr);

    // 3. 释放 C++ 侧的中转内存 (生命周期交接完毕)
    delete[] model_data;

    if (ret != RKNN_SUCC) {
        std::cerr << "rknn_init failed! ret = " << ret << std::endl;
        return -1;
    }
    std::cout << "RKNN init success! Context ID: " << ctx << std::endl;

    // 4. RKNN 资源销毁
    ret = rknn_destroy(ctx);
    if (ret != RKNN_SUCC) {
        std::cerr << "FATAL: RKNN context destroy failed! Error code: " << ret << std::endl;
    }

    return 0;
}
