#pragma once

#include <cstdint>
#include <string>
#include <vector>
#include "rknn_api.h"

struct RknnRunTiming {
    double input_copy_us = 0.0;
    double inputs_set_us = 0.0;
    double run_call_us = 0.0;
    double outputs_get_us = 0.0;
    double output_pack_us = 0.0;
    double outputs_release_us = 0.0;
    double total_us = 0.0;
};

class RknnEngine {
public:
    RknnEngine();
    ~RknnEngine();

    int init(const std::string& model_path);
    int input_setting(rknn_tensor_type type);
    int run(
        void* input_data,
        std::vector<float>& output_data,
        RknnRunTiming* timing = nullptr
    );

    uint32_t get_input_num() const { return io_num_.n_input; }
    uint32_t get_output_num() const { return io_num_.n_output; }

    const rknn_tensor_attr& get_input_attr(uint32_t idx = 0) const;
    const rknn_tensor_attr& get_output_attr(uint32_t idx = 0) const;
    uint32_t get_output_elem_num(uint32_t idx = 0) const;
    void print_model_tensor_info() const;

    // 解析输入 H/W/C（支持 NHWC/NCHW）；不支持的 fmt 返回 false。
    bool get_input_hw_c(uint32_t idx, uint32_t& h, uint32_t& w, uint32_t& c) const;

    // 推导 YOLO 输出布局：split 双输出 [1,4,N]+[1,C,N] 或单输出 [1,C,N]。
    // 成功时写入 field_count / candidate_count；无法识别返回 false。
    bool get_output_layout(uint32_t& field_count, uint32_t& candidate_count) const;

private:
    unsigned char* load_model_file(const std::string& path, int* size);

    rknn_context ctx_;
    bool is_initialized_;
    rknn_input_output_num io_num_;

    std::vector<rknn_tensor_attr> input_attrs_;
    std::vector<rknn_tensor_attr> output_attrs_;

    // 动态 vector，不再静态定长数组
    std::vector<rknn_input>  input_list_;
    std::vector<rknn_output> output_list_;
};
