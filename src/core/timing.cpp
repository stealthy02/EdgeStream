#include "edgestream/core/timing.h"

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
