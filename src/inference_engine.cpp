#include "inference_engine.h"

#include <cstddef>
#include <stdexcept>
#include <utility>

namespace {

std::size_t element_count(const std::vector<int64_t>& shape)
{
    std::size_t count = 1;
    for (const int64_t dimension : shape) {
        if (dimension <= 0) {
            throw std::runtime_error(
                "Only fixed positive tensor dimensions are supported."
            );
        }
        count *= static_cast<std::size_t>(dimension);
    }
    return count;
}

}  // namespace

InferenceEngine::InferenceEngine(const std::string& model_path)
    : env_(ORT_LOGGING_LEVEL_WARNING, "edgestream"),
      session_options_(),
      session_(env_, model_path.c_str(), session_options_),
      memory_info_(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault))
{
    size_t input_count = session_.GetInputCount();
    size_t output_count = session_.GetOutputCount();

    // 强制：单输入
    if(input_count != 1)
    {
        throw std::runtime_error("Only single input model supported");
    }

    Ort::AllocatorWithDefaultOptions allocator;

    // 加载输入名称（仅1个）
    auto in_name_owned = session_.GetInputNameAllocated(0, allocator);
    input_names_.push_back(in_name_owned.get());
    name_storage_.push_back(std::move(in_name_owned));

    // 加载全部输出名称，支持多输出
    for(size_t i = 0; i < output_count; i++){
        auto name_owned = session_.GetOutputNameAllocated(i, allocator);
        output_names_.push_back(name_owned.get());
        name_storage_.push_back(std::move(name_owned));
    }

    // 单输入shape
    input_shape_ = session_
        .GetInputTypeInfo(0)
        .GetTensorTypeAndShapeInfo()
        .GetShape();

    // 多输出shape + 缓存
    output_shapes_.reserve(output_count);
    output_caches_.reserve(output_count);
    for(size_t i = 0; i < output_count; i++){
        auto shape = session_
            .GetOutputTypeInfo(i)
            .GetTensorTypeAndShapeInfo()
            .GetShape();
        output_shapes_.push_back(shape);

        InferenceOutput cache;
        cache.shape = shape;
        const std::size_t expected_count = element_count(shape);
        cache.data.resize(expected_count);
        output_caches_.push_back(std::move(cache));
    }
}

const std::vector<const char*>& InferenceEngine::input_name() const noexcept
{
    return input_names_;
}

const std::vector<const char*>& InferenceEngine::output_name() const noexcept
{
    return output_names_;
}

const std::vector<int64_t>& InferenceEngine::input_shape() const noexcept
{
    return input_shape_;
}

const std::vector<std::vector<int64_t>>& InferenceEngine::output_shapes() const noexcept
{
    return output_shapes_;
}

const std::vector<InferenceOutput>& InferenceEngine::run(const std::vector<float>& input_data)
{
    const std::size_t expected_input_count = element_count(input_shape_);
    if (input_data.size() != expected_input_count) {
        throw std::invalid_argument(
            "Input element count does not match the model input shape."
        );
    }

    // 单输入tensor
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info_,
        const_cast<float*>(input_data.data()),
        input_data.size(),
        input_shape_.data(),
        input_shape_.size()
    );

    // 多输出tensor，绑定缓存
    std::vector<Ort::Value> output_tensors;
    output_tensors.reserve(output_names_.size());
    for(size_t i = 0; i < output_names_.size(); i++){
        auto& cache = output_caches_[i];
        output_tensors.push_back(Ort::Value::CreateTensor<float>(
            memory_info_,
            cache.data.data(),
            cache.data.size(),
            cache.shape.data(),
            cache.shape.size()
        ));
    }

    session_.Run(
        Ort::RunOptions{nullptr},
        input_names_.data(),
        &input_tensor,
        1,
        output_names_.data(),
        output_tensors.data(),
        output_tensors.size()
    );

    return output_caches_;
}
