#pragma once

#include <string>
#include <unordered_map>

// YOLO11 / COCO 80 类名映射（class_id 0-79）。
const std::unordered_map<int, std::string>& coco_class_names();

// class_id → 类名；未知返回 "class_<id>"。
std::string get_class_name(int class_id);
