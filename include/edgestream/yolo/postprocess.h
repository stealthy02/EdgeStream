#pragma once

#include <cstddef>
#include <vector>
#include "edgestream/yolo/preprocess.h"
struct Candidate {
    float cx{};
    float cy{};
    float w{};
    float h{};
    float score{};
    int class_id{-1};
};

struct Rectangle{
    float x1{};
    float y1{};
    float x2{};
    float y2{};
};


struct Detection {
    Rectangle rectangle{};
    float score{};
    int class_id{-1};
};

struct PostprocessTiming {
    double filter_us = 0.0;
    double convert_us = 0.0;
    double nms_us = 0.0;
    double restore_us = 0.0;
    double total_us = 0.0;
    std::size_t filtered_count = 0;
    std::size_t converted_count = 0;
    std::size_t kept_count = 0;
};

float calculate_rectangle_area(const Rectangle& rectangle);

float compute_iou(const Rectangle& a, const Rectangle& b);



// 这是第一步过滤掉置信度低的
std::vector<Candidate> filter_candidates(
    const float* output_data,
    int field_count,
    int candidate_count,
    float confidence_threshold);



// 这是第二步还原成xyxy
std::vector<Detection> convert_xywh_to_xyxy(
    const std::vector<Candidate>& candidates);

// 第三步 nms
std::vector<Detection> nms(const std::vector<Detection>& detections, float iou_threshold);

// 这是第四步还原成原先坐标
std::vector<Detection> restore_to_original(
    const std::vector<Detection>& detections,
    PreprocessParameter preprocess_parameter
);

std::vector<Detection> postprocess_image(
    const float* output_data,
    int field_count,
    int candidate_count,
    float confidence_threshold,
    float iou_threshold,
    PreprocessParameter preprocess_parameter,
    PostprocessTiming* timing = nullptr
);
