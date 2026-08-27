#include "postprocess.h"
#include <cmath>
#include <chrono>
#include <limits>
#include <vector>
#include <algorithm>
#include <map>

// 从output_data中筛选出满足阈值的预测框, 不包括loU和NMS
std::vector<Candidate> filter_candidates(
    const float* output_data,
    int field_count,
    int candidate_count,
    float confidence_threshold)
{
    std::vector<Candidate> filtered_candidates;
    // 预留容量，减少 push_back 时重新分配内存
    filtered_candidates.reserve(candidate_count);

    std::map<int, std::pair<float,int>> m;
    for(int i = candidate_count * 4; i < candidate_count * field_count;i++){
        const float score = output_data[i];
        if(score >= confidence_threshold){
            const int candidate_id = i % candidate_count;
            auto it = m.find(candidate_id);
            if(it == m.end() || it->second.first < score){
                m[candidate_id] = {score, i / candidate_count};
            }
        }
    }
    for (auto iter = m.begin(); iter != m.end(); ++iter){
        Candidate candidate;
        int candidate_id = iter->first;
        auto& pr = iter->second; // pr就是pair<float,int>

        candidate.cx = output_data[candidate_count * 0 + candidate_id];
        candidate.cy = output_data[candidate_count * 1 + candidate_id];
        candidate.w  = output_data[candidate_count * 2 + candidate_id];
        candidate.h  = output_data[candidate_count * 3 + candidate_id];
        candidate.score = pr.first;
        candidate.class_id = pr.second - 4;
        filtered_candidates.push_back(candidate);
    }

    return filtered_candidates;
}

// 从xywh转成xyxy的格式
std::vector<Detection> convert_xywh_to_xyxy(
    const std::vector<Candidate>& candidates)
{
    std::vector<Detection> detections;
    detections.reserve(candidates.size());

    for (const Candidate& candidate : candidates) {
        Detection detection;
        detection.rectangle.x1 = candidate.cx - candidate.w/2;
        detection.rectangle.y1 = candidate.cy - candidate.h/2;
        detection.rectangle.x2 = candidate.cx + candidate.w/2;
        detection.rectangle.y2 = candidate.cy + candidate.h/2;
        detection.score = candidate.score;
        detection.class_id = candidate.class_id;
        detections.push_back(detection);
    }

    return detections;
}

float calculate_rectangle_area(const Rectangle& rectangle)
{
    float w = std::max(0.0F, rectangle.x2 - rectangle.x1);
    float h = std::max(0.0F, rectangle.y2 - rectangle.y1);

    return w * h;
}

float compute_iou(const Rectangle& a, const Rectangle& b)
{
    Rectangle intersection;

    intersection.x1 = std::max(a.x1, b.x1);
    intersection.y1 = std::max(a.y1, b.y1);
    intersection.x2 = std::min(a.x2, b.x2);
    intersection.y2 = std::min(a.y2, b.y2);

    const float intersection_area =
        calculate_rectangle_area(intersection);

    if (intersection_area <= 0.0F) {
        return 0.0F;
    }

    const float area_a = calculate_rectangle_area(a);
    const float area_b = calculate_rectangle_area(b);

    const float union_area =
        area_a + area_b - intersection_area;

    if (union_area <= 0.0F) {
        return 0.0F;
    }

    return intersection_area / union_area;
}

std::vector<Detection> nms(
    const std::vector<Detection>& detections,
    float iou_threshold)
{
    std::vector<Detection> sorted_detections = detections;
    // 分数从高到低排, 方便删除同类iou过高的低分的
    std::sort(
        sorted_detections.begin(),
        sorted_detections.end(),
        [](const Detection& a, const Detection& b) {
            return a.score > b.score;
        });

    // 相当于 mask 不能真删, 直接用掩码就行
    std::vector<bool> suppressed(
        sorted_detections.size(),
        false
    );

    // 最终保留的
    std::vector<Detection> kept_detections;

    
    for (std::size_t i = 0; i < sorted_detections.size(); ++i) {
        if(suppressed[i]) continue;
        // 如果当前已经到这了(从高到低按iou删除后, 如果遍历到这还保留,说明肯定保留(因为分比这个高的都已经处理完了, 下面比的都比这个分低, 100%会保留))
        kept_detections.push_back(sorted_detections[i]);
        for (std::size_t j = i + 1; j < sorted_detections.size();++j) {
            if(suppressed[j]) continue;
            if(sorted_detections[i].class_id != sorted_detections[j].class_id) continue;
            if(compute_iou(sorted_detections[i].rectangle,sorted_detections[j].rectangle) > iou_threshold){
                suppressed[j] = true;
            }
        }
    }

    return kept_detections;
}

std::vector<Detection> restore_to_original(
    const std::vector<Detection>& detections,
    PreprocessParameter preprocess_parameter
){
    std::vector<Detection> detections_original(detections.size());

    for(std::size_t i = 0;i < detections.size(); i++){
        detections_original[i].rectangle.x1  = std::clamp((detections[i].rectangle.x1 - preprocess_parameter.pad_left) / preprocess_parameter.scale,0.0f,float(preprocess_parameter.original_width));
        detections_original[i].rectangle.x2  = std::clamp((detections[i].rectangle.x2 - preprocess_parameter.pad_left) / preprocess_parameter.scale,0.0f,float(preprocess_parameter.original_width));
        detections_original[i].rectangle.y1  = std::clamp((detections[i].rectangle.y1 - preprocess_parameter.pad_top) / preprocess_parameter.scale,0.0f,float(preprocess_parameter.original_height));
        detections_original[i].rectangle.y2  = std::clamp((detections[i].rectangle.y2 - preprocess_parameter.pad_top) / preprocess_parameter.scale,0.0f,float(preprocess_parameter.original_height));
        detections_original[i].class_id = detections[i].class_id;
        detections_original[i].score = detections[i].score;
    }
    return detections_original;
}


std::vector<Detection> postprocess_image(
    const float* output_data,
    int field_count,
    int candidate_count,
    float confidence_threshold,
    float iou_threshold,
    PreprocessParameter preprocess_parameter,
    PostprocessTiming* timing
){
    using clock = std::chrono::steady_clock;
    const auto now_if_timed = [timing]() {
        return timing != nullptr ? clock::now() : clock::time_point{};
    };
    const auto total_begin = now_if_timed();
    if (timing != nullptr) {
        *timing = {};
    }

    auto stage_begin = now_if_timed();
    std::vector<Candidate> filtered_candidates = filter_candidates(output_data,field_count,candidate_count,confidence_threshold);
    if (timing != nullptr) {
        timing->filter_us = std::chrono::duration<double, std::micro>(
            clock::now() - stage_begin).count();
        timing->filtered_count = filtered_candidates.size();
    }

    stage_begin = now_if_timed();
    const std::vector<Detection> detections_before_nms = convert_xywh_to_xyxy(filtered_candidates);
    if (timing != nullptr) {
        timing->convert_us = std::chrono::duration<double, std::micro>(
            clock::now() - stage_begin).count();
        timing->converted_count = detections_before_nms.size();
    }

    stage_begin = now_if_timed();
    const std::vector<Detection> detections_after_nms =  nms(detections_before_nms, iou_threshold);
    if (timing != nullptr) {
        timing->nms_us = std::chrono::duration<double, std::micro>(
            clock::now() - stage_begin).count();
        timing->kept_count = detections_after_nms.size();
    }

    stage_begin = now_if_timed();
    auto detections_original = restore_to_original(
        detections_after_nms,
        preprocess_parameter);
    if (timing != nullptr) {
        timing->restore_us = std::chrono::duration<double, std::micro>(
            clock::now() - stage_begin).count();
        timing->total_us = std::chrono::duration<double, std::micro>(
            clock::now() - total_begin).count();
    }
    return detections_original;
}
