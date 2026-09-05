#include "edgestream/core/timing.h"

#include <algorithm>
#include <cmath>

double calc_std(const std::vector<double>& data, double mean)
{
    if (data.size() <= 1) {
        return 0.0;
    }
    double sum_sq = 0.0;
    for (double v : data) {
        const double d = v - mean;
        sum_sq += d * d;
    }
    return std::sqrt(sum_sq / (data.size() - 1));
}

double mean_of(const std::vector<double>& data)
{
    double total = 0.0;
    for (double v : data) {
        total += v;
    }
    return data.empty() ? 0.0 : total / data.size();
}

double percentile_of(std::vector<double> data, double q)
{
    if (data.empty()) {
        return 0.0;
    }
    std::sort(data.begin(), data.end());
    q = std::clamp(q, 0.0, 1.0);
    const double index = q * static_cast<double>(data.size() - 1);
    const std::size_t lower = static_cast<std::size_t>(std::floor(index));
    const std::size_t upper = static_cast<std::size_t>(std::ceil(index));
    const double fraction = index - static_cast<double>(lower);
    return data[lower] + (data[upper] - data[lower]) * fraction;
}
