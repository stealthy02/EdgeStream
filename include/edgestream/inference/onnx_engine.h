#pragma once

#include <onnxruntime_cxx_api.h>
#include <cstddef>
#include <string>
#include <vector>

struct InferenceOutput
{
    std::vector<int64_t> shape;
    std::vector<float> data;
};

// ONNX Runtime 推理封装（单输入、多输出）。
class OnnxEngine
{
public:
    explicit OnnxEngine(const std::string& model_path);

    const std::vector<const char*>& input_name() const noexcept;
    const std::vector<const char*>& output_name() const noexcept;

    const std::vector<int64_t>& input_shape() const noexcept;
    const std::vector<std::vector<int64_t>>& output_shapes() const noexcept;

    const std::vector<InferenceOutput>& run(const std::vector<float>& input_data);

    // 合并 split 多输出为单个 [1, sum_dim1, dim2] 张量；单输出原样返回，空输入返回空。
    static InferenceOutput merge_split_outputs(std::vector<InferenceOutput> outputs);

private:
    Ort::Env env_;
    Ort::SessionOptions session_options_;
    Ort::Session session_;
    Ort::MemoryInfo memory_info_;

    std::vector<const char*> input_names_;
    std::vector<const char*> output_names_;
    std::vector<Ort::AllocatedStringPtr> name_storage_;

    std::vector<int64_t> input_shape_;
    std::vector<std::vector<int64_t>> output_shapes_;
    std::vector<InferenceOutput> output_caches_;
};
