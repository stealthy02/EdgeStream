#pragma once
#include <string>
#include "rknn_api.h"
#include "postprocess.h"
#include <vector>

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

private:
    unsigned char* load_model_file(const std::string& path, int* size);

    rknn_context ctx_;
    bool is_initialized_;
    rknn_input_output_num io_num_;

    std::vector<rknn_tensor_attr> input_attrs_;
    std::vector<rknn_tensor_attr> output_attrs_;

    // ✅ 动态vector，不再静态定长数组，无自定义宏
    std::vector<rknn_input>  input_list_;
    std::vector<rknn_output> output_list_;
};
