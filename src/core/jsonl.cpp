#include "edgestream/core/jsonl.h"

#include <fstream>
#include <string>

#include "edgestream/core/coco_names.h"
#include "edgestream/core/file_io.h"
#include <nlohmann/json.hpp>

bool save_detections_to_jsonl(
    const std::vector<Detection>& detections,
    const std::string& save_path)
{
    const auto last_slash = save_path.find_last_of('/');
    if (last_slash != std::string::npos) {
        const std::string dir = save_path.substr(0, last_slash);
        if (!mkdir_if_not_exist(dir)) {
            std::cerr << "❌ 创建目录失败: " << dir << std::endl;
            return false;
        }
    }

    std::ofstream fout(save_path);
    if (!fout.is_open()) {
        std::cerr << "❌ 打开文件失败: " << save_path << std::endl;
        return false;
    }

    for (std::size_t idx = 0; idx < detections.size(); ++idx) {
        const auto& det = detections[idx];
        nlohmann::json j;
        j["detection_index"] = static_cast<int>(idx);
        j["class_id"] = det.class_id;
        j["class_name"] = get_class_name(det.class_id);
        j["confidence"] = det.score;
        j["xyxy"] = nlohmann::json::array({
            det.rectangle.x1, det.rectangle.y1,
            det.rectangle.x2, det.rectangle.y2
        });
        fout << j.dump() << "\n";
    }
    return true;
}
