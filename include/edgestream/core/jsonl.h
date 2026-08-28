#pragma once

#include <string>
#include <vector>

#include "edgestream/yolo/postprocess.h"  // Detection

// 把检测结果逐行写为 JSONL；自动创建父目录。成功返回 true。
bool save_detections_to_jsonl(
    const std::vector<Detection>& detections,
    const std::string& save_path);
