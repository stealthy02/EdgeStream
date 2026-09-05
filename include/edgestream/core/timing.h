#pragma once

#include <vector>

// 样本标准差（数据量 <= 1 时返回 0）。
double calc_std(const std::vector<double>& data, double mean);

// 算术均值；空集合返回 0。
double mean_of(const std::vector<double>& data);

// 线性插值百分位数；q 取 [0, 1]，空集合返回 0。
double percentile_of(std::vector<double> data, double q);
