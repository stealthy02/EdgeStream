#include "edgestream/core/file_io.h"

#include <filesystem>

bool mkdir_if_not_exist(const std::string& dir)
{
    if (dir.empty()) {
        return false;
    }
    std::error_code ec;
    std::filesystem::create_directories(dir, ec);
    return !ec;
}
