#pragma once

#include <string>

// 递归创建目录（若已存在视为成功）。返回是否成功。
bool mkdir_if_not_exist(const std::string& dir);
