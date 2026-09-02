#include "edgestream/yolo/preprocess.h"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <chrono>
#include <stdexcept>
#include <vector>

static uint16_t float_to_fp16(float f)
{
    uint32_t bits;
    std::memcpy(&bits, &f, sizeof(float));

    uint32_t sign = bits >> 31;
    uint32_t exp  = (bits >> 23) & 0xFF;
    uint32_t mant = bits & 0x7FFFFF;

    uint16_t fp16_sign = static_cast<uint16_t>(sign);
    uint16_t fp16_exp, fp16_mant;

    if (exp == 0)
    {
        // 零 / 非规格化
        fp16_exp = 0;
        fp16_mant = static_cast<uint16_t>(mant >> 13);
    }
    else if (exp == 0xFF)
    {
        // INF / NaN
        fp16_exp = 0x1F;
        fp16_mant = static_cast<uint16_t>(mant >> 13);
    }
    else
    {
        int new_exp = static_cast<int>(exp) - 127 + 15;
        if (new_exp >= 0x1F)
        {
            // 溢出为inf
            fp16_exp = 0x1F;
            fp16_mant = 0;
        }
        else if (new_exp <= 0)
        {
            // 下溢，转为非规格化
            mant |= 0x800000;
            mant >>= (1 - new_exp);
            fp16_exp = 0;
            fp16_mant = static_cast<uint16_t>(mant >> 13);
        }
        else
        {
            fp16_exp = static_cast<uint16_t>(new_exp);
            fp16_mant = static_cast<uint16_t>(mant >> 13);
        }
    }
    return static_cast<uint16_t>((fp16_sign << 15) | (fp16_exp << 10) | fp16_mant);
}

// ===================== 纯FP32 实现 =====================
static void convert_NCHW(
    const cv::Mat& fulled_image,
    int input_width,
    int input_height,
    std::vector<float>& tensor_data)
{
    for (int y = 0; y < input_height; ++y) {
        for (int x = 0; x < input_width; ++x) {
            cv::Vec3b pixel = fulled_image.at<cv::Vec3b>(y, x);
            const float r = pixel[2] / 255.0F;
            const float g = pixel[1] / 255.0F;
            const float b = pixel[0] / 255.0F;

            const int index = y * input_width + x;
            tensor_data[0 * input_height * input_width + index] = r;
            tensor_data[1 * input_height * input_width + index] = g;
            tensor_data[2 * input_height * input_width + index] = b;
        }
    }
}

static void convert_NHWC(
    const cv::Mat& fulled_image,
    int input_width,
    int input_height,
    std::vector<float>& tensor_data)
{
    for (int y = 0; y < input_height; ++y) {
        for (int x = 0; x < input_width; ++x) {
            cv::Vec3b pixel = fulled_image.at<cv::Vec3b>(y, x);
            const float r = pixel[2] / 255.0F;
            const float g = pixel[1] / 255.0F;
            const float b = pixel[0] / 255.0F;

            const int index = y * input_width + x;
            tensor_data[index * 3 + 0] = r;
            tensor_data[index * 3 + 1] = g;
            tensor_data[index * 3 + 2] = b;
        }
    }
}

// ===================== 纯FP16 实现 =====================
static void convert_NCHW(
    const cv::Mat& fulled_image,
    int input_width,
    int input_height,
    std::vector<uint16_t>& tensor_data)
{
    for (int y = 0; y < input_height; ++y) {
        for (int x = 0; x < input_width; ++x) {
            cv::Vec3b pixel = fulled_image.at<cv::Vec3b>(y, x);
            const float r = pixel[2] / 255.0F;
            const float g = pixel[1] / 255.0F;
            const float b = pixel[0] / 255.0F;

            const int index = y * input_width + x;
            tensor_data[0 * input_height * input_width + index] = float_to_fp16(r);
            tensor_data[1 * input_height * input_width + index] = float_to_fp16(g);
            tensor_data[2 * input_height * input_width + index] = float_to_fp16(b);
        }
    }
}

static void convert_NHWC(
    const cv::Mat& fulled_image,
    int input_width,
    int input_height,
    std::vector<uint16_t>& tensor_data)
{
    for (int y = 0; y < input_height; ++y) {
        for (int x = 0; x < input_width; ++x) {
            cv::Vec3b pixel = fulled_image.at<cv::Vec3b>(y, x);
            const float r = pixel[2] / 255.0F;
            const float g = pixel[1] / 255.0F;
            const float b = pixel[0] / 255.0F;

            const int index = y * input_width + x;
            tensor_data[index * 3 + 0] = float_to_fp16(r);
            tensor_data[index * 3 + 1] = float_to_fp16(g);
            tensor_data[index * 3 + 2] = float_to_fp16(b);
        }
    }
}
// int8
inline int8_t int_to_int8(int fp)
{
    const int8_t in_zp   = 0;
    int q = fp + in_zp;
    // 四舍五入
    int32_t val = static_cast<int32_t>(nearbyint(q));
    // 钳位 int8 范围 [-128, 127]
    if(val < 0) val = 0;
    if(val > 127)  val = 255;
    // std::cout<<val<<" ";
    return static_cast<int8_t>(val);
}
static void convert_NCHW(
    const cv::Mat& fulled_image,
    int input_width,
    int input_height,
    std::vector<int8_t>& tensor_data)
{
    for (int y = 0; y < input_height; ++y) {
        for (int x = 0; x < input_width; ++x) {
            cv::Vec3b pixel = fulled_image.at<cv::Vec3b>(y, x);
            const int r = pixel[2];
            const int g = pixel[1];
            const int b = pixel[0];

            const int index = y * input_width + x;
            tensor_data[0 * input_height * input_width + index] = int_to_int8(r);
            tensor_data[1 * input_height * input_width + index] = int_to_int8(g);
            tensor_data[2 * input_height * input_width + index] = int_to_int8(b);
        }
    }
}

static void convert_NHWC(
    const cv::Mat& fulled_image,
    int input_width,
    int input_height,
    std::vector<int8_t>& tensor_data)
{
    for (int y = 0; y < input_height; ++y) {
        // 获取当前行的输入指针
        const cv::Vec3b* src = fulled_image.ptr<cv::Vec3b>(y);
        // 获取当前行的输出起始指针
        int8_t* dst = tensor_data.data() + y * input_width * 3;
        for (int x = 0; x < input_width; ++x) {
            dst[0] = static_cast<int8_t>(static_cast<int>(src[x][2]) - 128);
            dst[1] = static_cast<int8_t>(static_cast<int>(src[x][1]) - 128);
            dst[2] = static_cast<int8_t>(static_cast<int>(src[x][0]) - 128);
            dst += 3;
        }
    }
}

static void convert_NHWC_legacy(
    const cv::Mat& fulled_image,
    int input_width,
    int input_height,
    std::vector<int8_t>& tensor_data)
{
    for (int y = 0; y < input_height; ++y) {
        for (int x = 0; x < input_width; ++x) {
            cv::Vec3b pixel = fulled_image.at<cv::Vec3b>(y, x);
            const int r = pixel[2];
            const int g = pixel[1];
            const int b = pixel[0];

            const int index = y * input_width + x;
            tensor_data[index * 3 + 0] = int_to_int8(r);
            tensor_data[index * 3 + 1] = int_to_int8(g);
            tensor_data[index * 3 + 2] = int_to_int8(b);
        }
    }
}

PreprocessParameter get_preprocess_parameter(
    int original_width,
    int original_height,
    int input_width,
    int input_height)
{

    float scale = std::min(input_width/float(original_width), input_height/float(original_height));
    int resized_width = std::round(original_width * scale);
    int resized_height = std::round(original_height * scale);
    int p_w = input_width  - resized_width;
    int p_h = input_height - resized_height;
    int pad_left = p_w/2;
    int pad_right = p_w - pad_left;
    int pad_top = p_h/2;
    int pad_bottom = p_h - pad_top;

    PreprocessParameter result;
    result.original_width = original_width;
    result.original_height = original_height;
    result.scale = scale;
    result.pad_left = pad_left;
    result.pad_top = pad_top;
    result.pad_right = pad_right;
    result.pad_bottom = pad_bottom;

    return result;
}


void preprocess_image(
    const cv::Mat& bgr_image,
    const PreprocessParameter& preprocess_parameter,
    std::vector<float>& out_tensor_data,
    Tensor_format tensor_format)
{
    if(bgr_image.empty()){
        throw std::runtime_error("Image is empty.");
    }
    cv::Mat resized_image;
    int resized_width = std::round(bgr_image.cols * preprocess_parameter.scale);
    int resized_height = std::round(bgr_image.rows * preprocess_parameter.scale);
    cv::resize(bgr_image,resized_image,cv::Size(resized_width, resized_height));

    cv::Mat fulled_image;
    cv::copyMakeBorder(resized_image, fulled_image,
                       preprocess_parameter.pad_top, preprocess_parameter.pad_bottom,
                       preprocess_parameter.pad_left, preprocess_parameter.pad_right,
                       cv::BORDER_CONSTANT, cv::Scalar(114,114,114));

    int w = fulled_image.cols;
    int h = fulled_image.rows;
    switch (tensor_format)
    {
    case TENSOR_NCHW:
        convert_NCHW(fulled_image, w, h, out_tensor_data);
        break;
    case TENSOR_NHWC:
        convert_NHWC(fulled_image, w, h, out_tensor_data);
        break;
    default:
        break;
    }
}

// FP16 接口
void preprocess_image(
    const cv::Mat& bgr_image,
    const PreprocessParameter& preprocess_parameter,
    std::vector<uint16_t>& out_tensor_data,
    Tensor_format tensor_format)
{
    if(bgr_image.empty()){
        throw std::runtime_error("Image is empty.");
    }
    cv::Mat resized_image;
    int resized_width = std::round(bgr_image.cols * preprocess_parameter.scale);
    int resized_height = std::round(bgr_image.rows * preprocess_parameter.scale);
    cv::resize(bgr_image,resized_image,cv::Size(resized_width, resized_height));

    cv::Mat fulled_image;
    cv::copyMakeBorder(resized_image, fulled_image,
                       preprocess_parameter.pad_top, preprocess_parameter.pad_bottom,
                       preprocess_parameter.pad_left, preprocess_parameter.pad_right,
                       cv::BORDER_CONSTANT, cv::Scalar(114,114,114));

    int w = fulled_image.cols;
    int h = fulled_image.rows;
    switch (tensor_format)
    {
    case TENSOR_NCHW:
        convert_NCHW(fulled_image, w, h, out_tensor_data);
        break;
    case TENSOR_NHWC:
        convert_NHWC(fulled_image, w, h, out_tensor_data);
        break;
    default:
        break;
    }
}

// INT8 接口
void preprocess_image(
    const cv::Mat& bgr_image,
    const PreprocessParameter& preprocess_parameter,
    std::vector<int8_t>& out_tensor_data,
    Tensor_format tensor_format,
    PreprocessTiming* timing,
    Int8PackMode pack_mode)
{
    using clock = std::chrono::steady_clock;
    const auto now_if_timed = [timing]() {
        return timing != nullptr ? clock::now() : clock::time_point{};
    };
    const auto total_begin = now_if_timed();
    if (timing != nullptr) {
        *timing = {};
    }
    if(bgr_image.empty()){
        throw std::runtime_error("Image is empty.");
    }
    cv::Mat resized_image;
    int resized_width = std::round(bgr_image.cols * preprocess_parameter.scale);
    int resized_height = std::round(bgr_image.rows * preprocess_parameter.scale);
    auto stage_begin = now_if_timed();
    cv::resize(bgr_image,resized_image,cv::Size(resized_width, resized_height));
    if (timing != nullptr) {
        timing->resize_us = std::chrono::duration<double, std::micro>(
            clock::now() - stage_begin).count();
    }

    cv::Mat fulled_image;
    stage_begin = now_if_timed();
    cv::copyMakeBorder(resized_image, fulled_image,
                       preprocess_parameter.pad_top, preprocess_parameter.pad_bottom,
                       preprocess_parameter.pad_left, preprocess_parameter.pad_right,
                       cv::BORDER_CONSTANT, cv::Scalar(114,114,114));
    if (timing != nullptr) {
        timing->padding_us = std::chrono::duration<double, std::micro>(
            clock::now() - stage_begin).count();
    }

    stage_begin = now_if_timed();
    switch (tensor_format)
    {
    case TENSOR_NCHW:
        convert_NCHW(fulled_image, fulled_image.cols, fulled_image.rows, out_tensor_data);
        break;
    case TENSOR_NHWC:
        if (pack_mode == Int8PackMode::Legacy) {
            convert_NHWC_legacy(
                fulled_image, fulled_image.cols, fulled_image.rows, out_tensor_data);
        } else {
            convert_NHWC(
                fulled_image, fulled_image.cols, fulled_image.rows, out_tensor_data);
        }
        break;
    default:
        break;
    }
    if (timing != nullptr) {
        timing->pack_us = std::chrono::duration<double, std::micro>(
            clock::now() - stage_begin).count();
        timing->total_us = std::chrono::duration<double, std::micro>(
            clock::now() - total_begin).count();
    }
}
