#include "edgestream/core/timing.h"

#include <cmath>
#include <iostream>
#include <vector>

namespace {

bool expect_near(double actual, double expected, const char* label)
{
    if (std::abs(actual - expected) > 1e-9) {
        std::cerr << "FAIL: " << label << ": expected " << expected
                  << ", got " << actual << '\n';
        return false;
    }
    return true;
}

}  // namespace

int main()
{
    const std::vector<double> values{40.0, 10.0, 30.0, 20.0, 50.0};
    bool ok = true;
    ok &= expect_near(mean_of(values), 30.0, "mean");
    ok &= expect_near(percentile_of(values, 0.50), 30.0, "p50");
    ok &= expect_near(percentile_of(values, 0.95), 48.0, "p95");
    ok &= expect_near(percentile_of(values, 0.99), 49.6, "p99");
    ok &= expect_near(percentile_of({}, 0.95), 0.0, "empty percentile");
    return ok ? 0 : 1;
}
