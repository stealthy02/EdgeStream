#include "edgestream/inference/rknn_engine.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <cstring>
#include <cstdio>
#include <chrono>

RknnEngine::RknnEngine() : ctx_(0), is_initialized_(false)
{
    // vector 默认空，无需 memset
}

RknnEngine::~RknnEngine()
{
    if (is_initialized_)
    {
        // 释放每个 input 预先 malloc 的 buf
        for (auto& in : input_list_)
        {
            if (in.buf != nullptr)
            {
                free(in.buf);
                in.buf = nullptr;
            }
        }
        input_list_.clear();

        int ret = rknn_destroy(ctx_);
        if (ret != RKNN_SUCC)
        {
            std::cerr << "FATAL: RKNN context destroy failed! Error code: " << ret << std::endl;
        }
        is_initialized_ = false;
    }
}


int rknn_get_tensor_elem_byte_size(rknn_tensor_type type)
{
    switch (type)
    {
    case RKNN_TENSOR_FLOAT32:
        return 4;
    case RKNN_TENSOR_FLOAT16:
        return 2;
    case RKNN_TENSOR_INT8:
    case RKNN_TENSOR_UINT8:
    case RKNN_TENSOR_BOOL:
        return 1;
    case RKNN_TENSOR_INT16:
    case RKNN_TENSOR_UINT16:
    case RKNN_TENSOR_BFLOAT16:
        return 2;
    case RKNN_TENSOR_INT32:
    case RKNN_TENSOR_UINT32:
        return 4;
    case RKNN_TENSOR_INT64:
        return 8;
    case RKNN_TENSOR_INT4:
        /* int4 4bit，半个字节，int 无法返回小数，返回‑1 做特殊标记 */
        return -1;
    default:
        /* 未知类型返回0，代表错误 */
        return 0;
    }
}

int RknnEngine::init(const std::string& model_path)
{
    int model_size = 0;
    unsigned char* model_data = load_model_file(model_path, &model_size);
    if (!model_data)
    {
        std::cerr << "Failed to read model file." << std::endl;
        return -1;
    }

    ctx_ = 0;
    int ret = rknn_init(&ctx_, model_data, model_size, 0, nullptr);
    delete[] model_data;

    if (ret != RKNN_SUCC)
    {
        std::cerr << "rknn_init failed! Error code: " << ret << std::endl;
        return ret;
    }
    is_initialized_ = true;
    std::cout << "RKNN model loaded successfully." << std::endl;

    // 1. 查询真实输入输出数量
    ret = rknn_query(ctx_, RKNN_QUERY_IN_OUT_NUM, &io_num_, sizeof(io_num_));
    if (ret != RKNN_SUCC)
    {
        printf("rknn_query IN_OUT_NUM fail! ret=%d\n", ret);
        return -1;
    }

    // 2. 查询全部输入张量属性
    input_attrs_.resize(io_num_.n_input);
    for (uint32_t i = 0; i < io_num_.n_input; i++)
    {
        memset(&input_attrs_[i], 0, sizeof(rknn_tensor_attr));
        input_attrs_[i].index = i;
        ret = rknn_query(ctx_, RKNN_QUERY_INPUT_ATTR, &(input_attrs_[i]), sizeof(rknn_tensor_attr));
        if (ret != RKNN_SUCC)
        {
            printf("rknn_query input attr[%d] fail! ret=%d\n", i, ret);
            return -1;
        }
    }

    // 3. 查询全部输出张量属性
    output_attrs_.resize(io_num_.n_output);
    for (uint32_t i = 0; i < io_num_.n_output; i++)
    {
        memset(&output_attrs_[i], 0, sizeof(rknn_tensor_attr));
        output_attrs_[i].index = i;
        ret = rknn_query(ctx_, RKNN_QUERY_OUTPUT_ATTR, &(output_attrs_[i]), sizeof(rknn_tensor_attr));
        if (ret != RKNN_SUCC)
        {
            printf("rknn_query output attr[%d] fail! ret=%d\n", i, ret);
            return -1;
        }
    }


    output_list_.resize(io_num_.n_output);
    for (uint32_t i = 0; i < io_num_.n_output; i++)
    {
        auto& out = output_list_[i];
        memset(&out, 0, sizeof(rknn_output));

        out.index = i;
        out.want_float = 1;
        out.is_prealloc = 0;
    }
    return 0;
}

int RknnEngine::input_setting(rknn_tensor_type type){
    input_list_.resize(io_num_.n_input);
    for (uint32_t i = 0; i < io_num_.n_input; i++)
    {
        auto& in = input_list_[i];
        memset(&in, 0, sizeof(rknn_input));

        in.index = i;
        // Inputs are already quantized/converted and laid out by the caller.
        // Keep RKNN from applying an implicit conversion.
        in.pass_through = 1;
        in.type = type;
        in.fmt  = input_attrs_[i].fmt;
        in.size = input_attrs_[i].n_elems * rknn_get_tensor_elem_byte_size(in.type);
        in.buf  = malloc(in.size);

        if (in.buf == nullptr)
        {
            printf("分配输入内存失败! input index: %u\n", i);
            return -1;
        }
    }
    return 0;

}

int RknnEngine::run(
    void* input_data,
    std::vector<float>& output_data,
    RknnRunTiming* timing)
{
    using clock = std::chrono::steady_clock;
    const auto now_if_timed = [timing]() {
        return timing != nullptr ? clock::now() : clock::time_point{};
    };
    const auto total_begin = now_if_timed();
    if (timing != nullptr) {
        *timing = {};
    }

    if (!is_initialized_)
    {
        printf("引擎未初始化!\n");
        return -1;
    }
    if (input_data == nullptr)
    {
        printf("run input_data is nullptr!\n");
        return -1;
    }
    if (input_list_.empty())
    {
        printf("input list 为空!\n");
        return -1;
    }

    // This is the explicit staging copy in the current rknn_inputs_set path.
    auto stage_begin = now_if_timed();
    memcpy(input_list_[0].buf, input_data, input_list_[0].size);
    if (timing != nullptr) {
        timing->input_copy_us = std::chrono::duration<double, std::micro>(
            clock::now() - stage_begin).count();
    }

    stage_begin = now_if_timed();
    int ret = rknn_inputs_set(ctx_, io_num_.n_input, input_list_.data());
    if (timing != nullptr) {
        timing->inputs_set_us = std::chrono::duration<double, std::micro>(
            clock::now() - stage_begin).count();
    }
    if (ret < 0)
    {
        printf("rknn_inputs_set 失败! ret=%d\n", ret);
        return ret;
    }

    stage_begin = now_if_timed();
    ret = rknn_run(ctx_, nullptr);
    if (timing != nullptr) {
        timing->run_call_us = std::chrono::duration<double, std::micro>(
            clock::now() - stage_begin).count();
    }
    if (ret < 0)
    {
        printf("rknn_run 失败! ret=%d\n", ret);
        return ret;
    }

    stage_begin = now_if_timed();
    ret = rknn_outputs_get(ctx_, io_num_.n_output, output_list_.data(), nullptr);
    if (timing != nullptr) {
        timing->outputs_get_us = std::chrono::duration<double, std::micro>(
            clock::now() - stage_begin).count();
    }
    if (ret != RKNN_SUCC)
    {
        printf("rknn_outputs_get fail! ret=%d\n", ret);
        return ret;
    }

    // 兼容原始单输出和 split ONNX 的双输出：按输出顺序拼接成
    // [xywh(4), class_scores(80)]，这样现有 YOLO 后处理无需改变。
    stage_begin = now_if_timed();
    output_data.clear();
    for (uint32_t i = 0; i < io_num_.n_output; ++i)
    {
        uint32_t elem_count = output_attrs_[i].n_elems;
        const float* out_ptr = static_cast<const float*>(output_list_[i].buf);
        output_data.insert(output_data.end(), out_ptr, out_ptr + elem_count);
    }
    if (timing != nullptr) {
        timing->output_pack_us = std::chrono::duration<double, std::micro>(
            clock::now() - stage_begin).count();
    }

    stage_begin = now_if_timed();
    rknn_outputs_release(ctx_, io_num_.n_output, output_list_.data());
    if (timing != nullptr) {
        timing->outputs_release_us = std::chrono::duration<double, std::micro>(
            clock::now() - stage_begin).count();
        timing->total_us = std::chrono::duration<double, std::micro>(
            clock::now() - total_begin).count();
    }
    return 0;
}

unsigned char* RknnEngine::load_model_file(const std::string& path, int* size)
{
    std::ifstream file(path, std::ios::in | std::ios::binary | std::ios::ate);
    if (!file.is_open())
    {
        std::cerr << "Open file failed: " << path << std::endl;
        return nullptr;
    }
    *size = static_cast<int>(file.tellg());
    file.seekg(0, std::ios::beg);
    unsigned char* model_data = new unsigned char[*size];
    file.read(reinterpret_cast<char*>(model_data), *size);
    file.close();
    return model_data;
}

const rknn_tensor_attr& RknnEngine::get_input_attr(uint32_t idx) const
{
    return input_attrs_.at(idx);
}

const rknn_tensor_attr& RknnEngine::get_output_attr(uint32_t idx) const
{
    return output_attrs_.at(idx);
}

uint32_t RknnEngine::get_output_elem_num(uint32_t idx) const
{
    return output_attrs_.at(idx).n_elems;
}

bool RknnEngine::get_input_hw_c(uint32_t idx, uint32_t& h, uint32_t& w, uint32_t& c) const
{
    const rknn_tensor_attr& a = get_input_attr(idx);
    if (a.fmt == RKNN_TENSOR_NHWC) {
        h = a.dims[1]; w = a.dims[2]; c = a.dims[3];
        return true;
    }
    if (a.fmt == RKNN_TENSOR_NCHW) {
        h = a.dims[2]; w = a.dims[3]; c = a.dims[1];
        return true;
    }
    return false;
}

bool RknnEngine::get_output_layout(uint32_t& field_count, uint32_t& candidate_count) const
{
    if (get_output_num() == 2) {
        const rknn_tensor_attr& boxes = get_output_attr(0);
        const rknn_tensor_attr& scores = get_output_attr(1);
        if (boxes.n_dims != 3 || scores.n_dims != 3 ||
            boxes.dims[1] != 4 || boxes.dims[2] != scores.dims[2]) {
            return false;
        }
        field_count = boxes.dims[1] + scores.dims[1];
        candidate_count = boxes.dims[2];
        return true;
    }

    const rknn_tensor_attr& out = get_output_attr(0);
    if (out.n_dims != 3) {
        return false;
    }
    const uint32_t d1 = out.dims[1];
    const uint32_t d2 = out.dims[2];
    if (d1 == 84) {
        field_count = d1; candidate_count = d2;
        return true;
    }
    if (d2 == 84) {
        field_count = d2; candidate_count = d1;
        return true;
    }
    return false;
}

void RknnEngine::print_model_tensor_info() const
{
    auto get_tensor_fmt_str = [](rknn_tensor_format fmt) -> const char* {
        switch (fmt)
        {
            case RKNN_TENSOR_NCHW:      return "RKNN_TENSOR_NCHW";
            case RKNN_TENSOR_NHWC:      return "RKNN_TENSOR_NHWC";
            case RKNN_TENSOR_NC1HWC2:   return "RKNN_TENSOR_NC1HWC2";
            case RKNN_TENSOR_UNDEFINED: return "RKNN_TENSOR_UNDEFINED";
            default:                    return "UNKNOWN_FMT";
        }
    };

    auto get_tensor_type_str = [](rknn_tensor_type type) -> const char* {
        switch (type)
        {
            case RKNN_TENSOR_FLOAT32:   return "RKNN_TENSOR_FLOAT32";
            case RKNN_TENSOR_FLOAT16:  return "RKNN_TENSOR_FLOAT16";
            case RKNN_TENSOR_INT8:     return "RKNN_TENSOR_INT8";
            case RKNN_TENSOR_UINT8:    return "RKNN_TENSOR_UINT8";
            case RKNN_TENSOR_INT16:    return "RKNN_TENSOR_INT16";
            case RKNN_TENSOR_UINT16:   return "RKNN_TENSOR_UINT16";
            case RKNN_TENSOR_INT32:    return "RKNN_TENSOR_INT32";
            case RKNN_TENSOR_UINT32:   return "RKNN_TENSOR_UINT32";
            case RKNN_TENSOR_INT64:    return "RKNN_TENSOR_INT64";
            case RKNN_TENSOR_BOOL:     return "RKNN_TENSOR_BOOL";
            case RKNN_TENSOR_INT4:     return "RKNN_TENSOR_INT4";
            case RKNN_TENSOR_BFLOAT16: return "RKNN_TENSOR_BFLOAT16";
            default:                   return "UNKNOWN_TYPE";
        }
    };

    printf("\n===== Model Tensor Info =====\n");
    printf("Input num: %u, Output num: %u\n", io_num_.n_input, io_num_.n_output);


    for (size_t i = 0; i < input_attrs_.size(); i++)
    {
        const auto& attr = input_attrs_[i];
        printf("[Input %zu] name=%s, dims=[", i, attr.name);
        printf("Scale: %f, Zero Point: %d\n", input_attrs_[i].scale, input_attrs_[i].zp);
        for (uint32_t d = 0; d < attr.n_dims; d++)
        {
            printf("%u ", attr.dims[d]);
        }
        printf("], n_elems=%u, size(bytes)=%u, fmt=%s, type=%s\n",
               attr.n_elems, attr.size,
               get_tensor_fmt_str(static_cast<rknn_tensor_format>(attr.fmt)),
               get_tensor_type_str(static_cast<rknn_tensor_type>(attr.type)));
    }


    for (size_t i = 0; i < output_attrs_.size(); i++)
    {
        const auto& attr = output_attrs_[i];
        printf("[Output %zu] name=%s, dims=[", i, attr.name);
        for (uint32_t d = 0; d < attr.n_dims; d++)
        {
            printf("%u ", attr.dims[d]);
        }
        printf("], n_elems=%u, size(bytes)=%u, fmt=%s, type=%s\n",
               attr.n_elems, attr.size,
               get_tensor_fmt_str(static_cast<rknn_tensor_format>(attr.fmt)),
               get_tensor_type_str(static_cast<rknn_tensor_type>(attr.type)));
    }
    printf("=============================\n\n");
}
