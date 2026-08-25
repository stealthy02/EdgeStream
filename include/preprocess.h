#pragma once
#include <iostream>
#include <opencv2/core.hpp>
#include <vector>

enum Tensor_format{
    TENSOR_NCHW = 0,
    TENSOR_NHWC
};

struct PreprocessParameter {
    int original_width{};
    int original_height{};
    float scale{};
    int pad_left{};
    int pad_top{};
    int pad_right{};
    int pad_bottom{};
};

struct PreprocessTiming {
    double resize_us = 0.0;
    double padding_us = 0.0;
    double pack_us = 0.0;
    double total_us = 0.0;
};

PreprocessParameter get_preprocess_parameter(
    int original_width,
    int original_height,
    int input_width,
    int input_height);

void preprocess_image(
    const cv::Mat& bgr_image,
    const PreprocessParameter& preprocess_parameter,
    std::vector<float>& out_tensor_data,
    Tensor_format tensor_format
);

void preprocess_image(
    const cv::Mat& bgr_image,
    const PreprocessParameter& preprocess_parameter,
    std::vector<uint16_t>& out_tensor_data,
    Tensor_format tensor_format
);

void preprocess_image(
    const cv::Mat& bgr_image,
    const PreprocessParameter& preprocess_parameter,
    std::vector<int8_t>& out_tensor_data,
    Tensor_format tensor_format,
    PreprocessTiming* timing = nullptr
);
