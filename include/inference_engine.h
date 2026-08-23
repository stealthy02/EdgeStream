#pragma once

#include <onnxruntime_cxx_api.h>
#include <vector>
#include <string>

struct InferenceOutput
{
    std::vector<int64_t> shape;
    std::vector<float> data;
};

class InferenceEngine
{
public:
    explicit InferenceEngine(const std::string& model_path);

    const std::vector<const char*>& input_name() const noexcept;
    const std::vector<const char*>& output_name() const noexcept;

    const std::vector<int64_t>& input_shape() const noexcept;
    const std::vector<std::vector<int64_t>>& output_shapes() const noexcept;

    const std::vector<InferenceOutput>& run(const std::vector<float>& input_data);

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
