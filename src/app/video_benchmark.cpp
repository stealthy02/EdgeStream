// ============================================================================
// video_benchmark —— 统一的视频 300 帧基准（取代 onnx/rknn/rknn_int8 三个独立测试）
//
// 用法：
//   video_benchmark --backend onnx|rknn-fp16|rknn-int8 \
//                    --model <PATH> [--video <PATH>] [--output-dir <DIR>] \
//                    [--conf 0.25] [--iou 0.70]
//
// 输出（写入 --output-dir，默认按后端 artifacts/onnx | artifacts/rknn）：
//   <tag>_raw.jsonl    逐帧耗时 + 各阶段 breakdown + system 快照
//   <tag>_stats.json   高层 prep/infer/post/total 的均值、标准差、p50/p95/p99、处理 FPS
//   tag ∈ {onnx, rknn_fp16, rknn_int8}
//
// 阶段 breakdown 可用性：INT8 有预处理阶段；RKNN(FP16/INT8) 有推理阶段；后处理阶段所有后端都有。
// ============================================================================

#include <algorithm>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

#include <opencv2/opencv.hpp>
#include <nlohmann/json.hpp>

#include "edgestream/concurrency/thread_safe_queue.h"
#include "edgestream/core/file_io.h"
#include "edgestream/core/signal.h"
#include "edgestream/core/timing.h"
#include "edgestream/inference/onnx_engine.h"
#include "edgestream/inference/rknn_engine.h"
#include "edgestream/yolo/postprocess.h"
#include "edgestream/yolo/preprocess.h"

using json = nlohmann::json;
namespace fs = std::filesystem;

// ============================================================================
// 系统快照（温度 / CPU0 频率 / NPU 频率）—— 仅本工具使用
// ============================================================================
struct SystemSnapshot {
    double max_temperature_c = -1.0;
    double cpu0_frequency_mhz = -1.0;
    double npu_frequency_mhz = -1.0;
};

static bool read_numeric_file(const fs::path& path, double& value)
{
    std::ifstream input(path);
    if (!input.is_open()) {
        return false;
    }
    input >> value;
    return input.good() || input.eof();
}

static double read_temperature_celsius()
{
    double max_temperature_c = -1.0;
    std::error_code error;
    const fs::path thermal_root = "/sys/class/thermal";
    if (!fs::exists(thermal_root, error)) {
        return max_temperature_c;
    }
    for (const auto& entry : fs::directory_iterator(thermal_root, error)) {
        if (error || !entry.is_directory(error)) {
            continue;
        }
        const std::string name = entry.path().filename().string();
        if (name.rfind("thermal_zone", 0) != 0) {
            continue;
        }
        double raw_temperature = 0.0;
        if (!read_numeric_file(entry.path() / "temp", raw_temperature)) {
            continue;
        }
        const double temperature_c = raw_temperature > 1000.0
            ? raw_temperature / 1000.0
            : raw_temperature;
        max_temperature_c = std::max(max_temperature_c, temperature_c);
    }
    return max_temperature_c;
}

static double read_cpu0_frequency_mhz()
{
    double frequency_khz = 0.0;
    const fs::path cpufreq_root = "/sys/devices/system/cpu/cpu0/cpufreq";
    if (read_numeric_file(cpufreq_root / "scaling_cur_freq", frequency_khz) ||
        read_numeric_file(cpufreq_root / "cpuinfo_cur_freq", frequency_khz)) {
        return frequency_khz / 1000.0;
    }
    return -1.0;
}

static double read_npu_frequency_mhz()
{
    std::error_code error;
    const fs::path devfreq_root = "/sys/class/devfreq";
    if (!fs::exists(devfreq_root, error)) {
        return -1.0;
    }
    for (const auto& entry : fs::directory_iterator(devfreq_root, error)) {
        if (error) {
            continue;
        }
        std::string identity = entry.path().filename().string();
        std::ifstream name_file(entry.path() / "name");
        std::string device_name;
        if (name_file.is_open()) {
            std::getline(name_file, device_name);
            identity += " " + device_name;
        }
        std::transform(identity.begin(), identity.end(), identity.begin(),
                       [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
        if (identity.find("npu") == std::string::npos) {
            continue;
        }
        double frequency_hz = 0.0;
        if (read_numeric_file(entry.path() / "cur_freq", frequency_hz)) {
            return frequency_hz / 1000000.0;
        }
    }
    return -1.0;
}

static SystemSnapshot read_system_snapshot()
{
    return {read_temperature_celsius(), read_cpu0_frequency_mhz(), read_npu_frequency_mhz()};
}

// ============================================================================
// 每帧计时 + 累积统计
// ============================================================================
struct FrameTimings {
    double prep_ms = 0.0;
    double infer_ms = 0.0;
    double post_ms = 0.0;
    double total_ms = 0.0;

    PreprocessTiming prep_timing{};   // 仅 INT8 填充阶段
    bool has_prep_stages = false;

    RknnRunTiming run_timing{};       // 仅 RKNN 填充阶段
    bool has_infer_stages = false;

    PostprocessTiming post_timing{};  // 所有后端都填充
};

struct BenchmarkStats {
    int frame_cnt = 0;
    std::vector<double> prep, infer, post, total;
    double total_prep = 0.0, total_infer = 0.0, total_post = 0.0;
    bool has_prep_stages = false, has_infer_stages = false;
    std::vector<double> p_resize, p_padding, p_pack, p_total;
    std::vector<double> i_input_copy, i_inputs_set, i_run_call,
                        i_outputs_get, i_output_pack, i_outputs_release, i_total;
    std::vector<double> o_filter, o_convert, o_nms, o_restore, o_total,
                        o_filtered, o_kept;
    std::vector<double> temp, cpu_freq, npu_freq;
    std::vector<SystemSnapshot> system;

    void accumulate(const FrameTimings& t, int frame_index)
    {
        total_prep  += t.prep_ms;
        total_infer += t.infer_ms;
        total_post  += t.post_ms;
        prep.push_back(t.prep_ms);
        infer.push_back(t.infer_ms);
        post.push_back(t.post_ms);
        total.push_back(t.total_ms);

        if (t.has_prep_stages) {
            p_resize.push_back(t.prep_timing.resize_us);
            p_padding.push_back(t.prep_timing.padding_us);
            p_pack.push_back(t.prep_timing.pack_us);
            p_total.push_back(t.prep_timing.total_us);
        }
        if (t.has_infer_stages) {
            i_input_copy.push_back(t.run_timing.input_copy_us);
            i_inputs_set.push_back(t.run_timing.inputs_set_us);
            i_run_call.push_back(t.run_timing.run_call_us);
            i_outputs_get.push_back(t.run_timing.outputs_get_us);
            i_output_pack.push_back(t.run_timing.output_pack_us);
            i_outputs_release.push_back(t.run_timing.outputs_release_us);
            i_total.push_back(t.run_timing.total_us);
        }
        o_filter.push_back(t.post_timing.filter_us);
        o_convert.push_back(t.post_timing.convert_us);
        o_nms.push_back(t.post_timing.nms_us);
        o_restore.push_back(t.post_timing.restore_us);
        o_total.push_back(t.post_timing.total_us);
        o_filtered.push_back(static_cast<double>(t.post_timing.filtered_count));
        o_kept.push_back(static_cast<double>(t.post_timing.kept_count));

        SystemSnapshot snapshot;
        if (frame_index == 0 || (frame_index + 1) % 10 == 0) {
            snapshot = read_system_snapshot();
        }
        system.push_back(snapshot);
        if (snapshot.max_temperature_c >= 0.0) temp.push_back(snapshot.max_temperature_c);
        if (snapshot.cpu0_frequency_mhz  >= 0.0) cpu_freq.push_back(snapshot.cpu0_frequency_mhz);
        if (snapshot.npu_frequency_mhz    >= 0.0) npu_freq.push_back(snapshot.npu_frequency_mhz);
    }
};

// 单帧推理回调：返回 0 表示成功，非 0 表示失败（应停止）
using InferFn = std::function<int(const cv::Mat& frame, FrameTimings& t)>;

// ============================================================================
// 统计落盘
// ============================================================================
static void write_results(const BenchmarkStats& s,
                          const std::string& output_dir, const std::string& tag,
                          const std::string& model, const std::string& video,
                          float conf, float iou)
{
    mkdir_if_not_exist(output_dir);
    const std::string raw_path   = output_dir + "/" + tag + "_raw.jsonl";
    const std::string stats_path = output_dir + "/" + tag + "_stats.json";

    // ---- raw jsonl ----
    std::ofstream raw(raw_path);
    if (!raw.is_open()) {
        std::cerr << "\n❌ Failed to write raw timings to " << raw_path << std::endl;
    } else {
        for (std::size_t i = 0; i < s.total.size(); ++i) {
            json sample{{"frame", i + 1}, {"preprocess_ms", s.prep[i]},
                        {"infer_ms", s.infer[i]}, {"postprocess_ms", s.post[i]},
                        {"frame_total_ms", s.total[i]}};
            if (s.has_prep_stages) {
                sample["preprocess_stages_ms"] = {
                    {"resize", s.p_resize[i] / 1000.0},
                    {"padding", s.p_padding[i] / 1000.0},
                    {"pack", s.p_pack[i] / 1000.0},
                    {"total", s.p_total[i] / 1000.0}
                };
            }
            if (s.has_infer_stages) {
                sample["infer_stages_ms"] = {
                    {"input_copy", s.i_input_copy[i] / 1000.0},
                    {"inputs_set", s.i_inputs_set[i] / 1000.0},
                    {"run_call", s.i_run_call[i] / 1000.0},
                    {"outputs_get", s.i_outputs_get[i] / 1000.0},
                    {"output_pack", s.i_output_pack[i] / 1000.0},
                    {"outputs_release", s.i_outputs_release[i] / 1000.0},
                    {"total", s.i_total[i] / 1000.0}
                };
            }
            sample["postprocess_stages_ms"] = {
                {"filter", s.o_filter[i] / 1000.0},
                {"convert", s.o_convert[i] / 1000.0},
                {"nms", s.o_nms[i] / 1000.0},
                {"restore", s.o_restore[i] / 1000.0},
                {"total", s.o_total[i] / 1000.0},
                {"filtered_count", s.o_filtered[i]},
                {"kept_count", s.o_kept[i]}
            };
            sample["system"] = {
                {"max_temperature_c", s.system[i].max_temperature_c},
                {"cpu0_frequency_mhz", s.system[i].cpu0_frequency_mhz},
                {"npu_frequency_mhz", s.system[i].npu_frequency_mhz}
            };
            raw << sample.dump() << '\n';
        }
        std::cout << "Raw timings saved to: " << raw_path << std::endl;
    }

    // ---- stats json ----
    const double avg_prep  = s.total_prep / s.frame_cnt;
    const double avg_infer = s.total_infer / s.frame_cnt;
    const double avg_post  = s.total_post / s.frame_cnt;
    const double avg_total = (s.total_prep + s.total_infer + s.total_post) / s.frame_cnt;

    const double processing_fps = avg_total > 0.0 ? 1000.0 / avg_total : 0.0;
    const auto latency_summary = [](const std::vector<double>& values, double mean) {
        return json{{"mean", mean},
                    {"std", calc_std(values, mean)},
                    {"p50", percentile_of(values, 0.50)},
                    {"p95", percentile_of(values, 0.95)},
                    {"p99", percentile_of(values, 0.99)}};
    };

    std::cout << "\n===== Stat over " << s.frame_cnt << " frames =====" << std::endl;
    std::cout << std::fixed << std::setprecision(3)
              << "Frame total: mean=" << avg_total << " ms"
              << " p50=" << percentile_of(s.total, 0.50) << " ms"
              << " p95=" << percentile_of(s.total, 0.95) << " ms"
              << " p99=" << percentile_of(s.total, 0.99) << " ms\n"
              << "Processing FPS: " << processing_fps << std::endl;

    json stats;
    stats["frame_count"] = s.frame_cnt;
    stats["unit"] = "ms";
    stats["run"] = {{"backend", tag}, {"model", model}, {"video", video},
                    {"confidence_threshold", conf}, {"iou_threshold", iou}};
    stats["processing_fps"] = processing_fps;
    stats["preprocess"] = latency_summary(s.prep, avg_prep);
    stats["infer"] = latency_summary(s.infer, avg_infer);
    stats["postprocess"] = latency_summary(s.post, avg_post);
    stats["frame_total"] = latency_summary(s.total, avg_total);

    auto save_breakdown = [&stats](
        const char* category, const char* name, const std::vector<double>& values) {
        const double m = mean_of(values);
        stats["breakdown"][category][name]["mean_ms"] = m / 1000.0;
        stats["breakdown"][category][name]["std_ms"]  = calc_std(values, m) / 1000.0;
    };
    if (s.has_prep_stages) {
        save_breakdown("preprocess", "resize",  s.p_resize);
        save_breakdown("preprocess", "padding", s.p_padding);
        save_breakdown("preprocess", "pack",    s.p_pack);
        save_breakdown("preprocess", "total",   s.p_total);
    }
    if (s.has_infer_stages) {
        save_breakdown("infer", "input_copy",      s.i_input_copy);
        save_breakdown("infer", "inputs_set",      s.i_inputs_set);
        save_breakdown("infer", "run_call",        s.i_run_call);
        save_breakdown("infer", "outputs_get",     s.i_outputs_get);
        save_breakdown("infer", "output_pack",     s.i_output_pack);
        save_breakdown("infer", "outputs_release", s.i_outputs_release);
        save_breakdown("infer", "total",           s.i_total);
    }
    save_breakdown("postprocess", "filter",   s.o_filter);
    save_breakdown("postprocess", "convert",  s.o_convert);
    save_breakdown("postprocess", "nms",      s.o_nms);
    save_breakdown("postprocess", "restore",  s.o_restore);
    save_breakdown("postprocess", "total",    s.o_total);

    stats["postprocess"]["filtered_count"] = {
        {"mean", mean_of(s.o_filtered)}, {"std", calc_std(s.o_filtered, mean_of(s.o_filtered))}
    };
    stats["postprocess"]["kept_count"] = {
        {"mean", mean_of(s.o_kept)}, {"std", calc_std(s.o_kept, mean_of(s.o_kept))}
    };
    stats["system"] = {
        {"max_temperature_c",   {{"mean", mean_of(s.temp)},     {"std", calc_std(s.temp, mean_of(s.temp))}}},
        {"cpu0_frequency_mhz",  {{"mean", mean_of(s.cpu_freq)}, {"std", calc_std(s.cpu_freq, mean_of(s.cpu_freq))}}},
        {"npu_frequency_mhz",   {{"mean", mean_of(s.npu_freq)}, {"std", calc_std(s.npu_freq, mean_of(s.npu_freq))}}}
    };

    std::ofstream f(stats_path);
    if (f.is_open()) {
        f << stats.dump(4);
        f.close();
        std::cout << "\n✅ Stats json saved to: " << stats_path << std::endl;
    } else {
        std::cerr << "\n❌ Failed to write json to " << stats_path << std::endl;
    }
}

// ============================================================================
// 命令行参数
// ============================================================================
enum class Backend { Onnx, RknnFp16, RknnInt8 };

struct Args {
    Backend backend = Backend::Onnx;
    std::string backend_tag;          // 文件名用：onnx / rknn_fp16 / rknn_int8
    std::string model;
    std::string video = "assets/regression/input.mp4";
    std::string output_dir;           // 空则按后端默认
    float conf = 0.25f;
    float iou = 0.70f;
};

static void print_usage(const char* prog)
{
    std::cerr
        << "用法: " << prog << " --backend <onnx|rknn-fp16|rknn-int8> --model <PATH> [选项]\n\n"
        << "  --backend <B>      【必填】后端：onnx / rknn-fp16 / rknn-int8\n"
        << "  --model <PATH>      【必填】模型路径\n"
        << "  --video <PATH>      视频路径 (默认: assets/regression/input.mp4)\n"
        << "  --output-dir <DIR>  输出目录 (默认: artifacts/onnx | artifacts/rknn)\n"
        << "  --conf <float>      置信度阈值 (默认 0.25)\n"
        << "  --iou  <float>      NMS IoU 阈值 (默认 0.70)\n"
        << "  -h / --help         显示此帮助\n";
}

static Args parse_args(int argc, char* argv[])
{
    Args a;
    for (int i = 1; i < argc; ++i) {
        const std::string s = argv[i];
        auto need_next = [&](const char* name) -> std::string {
            if (i + 1 >= argc) {
                throw std::runtime_error(std::string("参数 ") + name + " 缺少值");
            }
            return std::string(argv[++i]);
        };
        if (s == "-h" || s == "--help")            { print_usage(argv[0]); exit(0); }
        else if (s == "--backend") {
            const std::string b = need_next("--backend");
            if (b == "onnx")                         { a.backend = Backend::Onnx;     a.backend_tag = "onnx"; }
            else if (b == "rknn-fp16" || b == "rknn_fp16") { a.backend = Backend::RknnFp16; a.backend_tag = "rknn_fp16"; }
            else if (b == "rknn-int8" || b == "rknn_int8") { a.backend = Backend::RknnInt8; a.backend_tag = "rknn_int8"; }
            else throw std::runtime_error("未知后端: " + b);
        }
        else if (s == "--model")                    { a.model = need_next("--model"); }
        else if (s == "--video")                    { a.video = need_next("--video"); }
        else if (s == "--output-dir")               { a.output_dir = need_next("--output-dir"); }
        else if (s == "--conf")                     { a.conf = std::stof(need_next("--conf")); }
        else if (s == "--iou")                      { a.iou  = std::stof(need_next("--iou")); }
        else {
            std::cerr << "⚠️  未知参数: " << s << "\n\n";
            print_usage(argv[0]);
            throw std::runtime_error("参数解析失败");
        }
    }
    if (a.backend_tag.empty() || a.model.empty()) {
        print_usage(argv[0]);
        throw std::runtime_error("--backend 与 --model 均为必填");
    }
    if (a.output_dir.empty()) {
        a.output_dir = (a.backend == Backend::Onnx) ? "artifacts/onnx" : "artifacts/rknn";
    }
    return a;
}

// ============================================================================
// 生产者：视频帧 → 队列
// ============================================================================
static ThreadSafeQueue<cv::Mat> inference_queue(5);
static PreprocessParameter preprocess_parameter;

static void preprocess_thread(cv::VideoCapture& cap)
{
    if (!cap.isOpened()) {
        std::cerr << "Failed to open video." << std::endl;
        inference_queue.stop();
        return;
    }
    cv::Mat frame;
    while (!is_stop_requested() && cap.read(frame)) {
        if (frame.empty()) continue;
        if (!inference_queue.push(frame)) {
            break;
        }
    }
    inference_queue.stop();
}

// ============================================================================
// 驱动循环：生产者线程 + 主线程消费回调 + 统计
// ============================================================================
static void run_video_benchmark(InferFn infer, const Args& args,
                                bool has_prep_stages, bool has_infer_stages)
{
    cv::VideoCapture cap(args.video);
    if (!cap.isOpened()) {
        std::cerr << "❌ 无法打开视频: " << args.video << std::endl;
        return;
    }
    std::thread producer(preprocess_thread, std::ref(cap));

    BenchmarkStats stats;
    stats.has_prep_stages = has_prep_stages;
    stats.has_infer_stages = has_infer_stages;

    cv::Mat frame;
    while (inference_queue.pop(frame)) {
        FrameTimings t;
        const int ret = infer(frame, t);
        if (ret != 0) {
            inference_queue.stop();
            const int attempted = stats.frame_cnt + 1;
            std::cerr << "inference failed at frame " << attempted << ", ret=" << ret << std::endl;
            break;
        }
        stats.accumulate(t, stats.frame_cnt);
        stats.frame_cnt++;
    }

    producer.join();
    cap.release();

    if (stats.frame_cnt <= 0) {
        std::cout << "\n[WARN] No frame processed, skip save json" << std::endl;
        std::cout << "\n[INFO] Program exit gracefully" << std::endl;
        return;
    }
    write_results(stats, args.output_dir, args.backend_tag, args.model, args.video,
                  args.conf, args.iou);
    std::cout << "\n[INFO] Program exit gracefully" << std::endl;
}

// ============================================================================
// main
// ============================================================================
int main(int argc, char* argv[])
{
    Args args;
    try {
        args = parse_args(argc, argv);
    } catch (const std::exception& e) {
        std::cerr << "❌ " << e.what() << std::endl;
        return 1;
    }

    if (!fs::exists(args.model)) {
        std::cerr << "❌ 模型不存在: " << args.model << std::endl;
        return 1;
    }

    install_sigint_handler();
    std::cout << "========================================" << std::endl;
    std::cout << "video_benchmark" << std::endl;
    std::cout << "  backend    = " << args.backend_tag << std::endl;
    std::cout << "  model      = " << args.model << std::endl;
    std::cout << "  video      = " << args.video << std::endl;
    std::cout << "  output_dir = " << args.output_dir << std::endl;
    std::cout << "  conf/iou   = " << args.conf << " / " << args.iou << std::endl;
    std::cout << "========================================" << std::endl;

    using clock = std::chrono::high_resolution_clock;
    const auto ms = [](clock::time_point a, clock::time_point b) {
        return std::chrono::duration<double, std::milli>(b - a).count();
    };

    if (args.backend == Backend::Onnx) {
        OnnxEngine engine(args.model);
        const auto input_shape = engine.input_shape();
        cv::VideoCapture probe(args.video);
        const int width  = probe.get(cv::CAP_PROP_FRAME_WIDTH);
        const int height = probe.get(cv::CAP_PROP_FRAME_HEIGHT);
        probe.release();
        preprocess_parameter = get_preprocess_parameter(width, height, input_shape[3], input_shape[2]);
        std::vector<float> tensor_data(3 * input_shape[2] * input_shape[3]);

        const float conf = args.conf, iou = args.iou;
        InferFn infer = [&, conf, iou, ms](const cv::Mat& f, FrameTimings& t) -> int {
            const auto a = clock::now();
            preprocess_image(f, preprocess_parameter, tensor_data, TENSOR_NCHW);
            const auto b = clock::now();
            auto outputs = engine.run(tensor_data);
            InferenceOutput merged = OnnxEngine::merge_split_outputs(std::move(outputs));
            const auto c = clock::now();
            PostprocessTiming pt;
            postprocess_image(merged.data.data(),
                              static_cast<int>(merged.shape[1]),
                              static_cast<int>(merged.shape[2]),
                              conf, iou, preprocess_parameter, &pt);
            const auto d = clock::now();
            t.prep_ms = ms(a, b); t.infer_ms = ms(b, c); t.post_ms = ms(c, d);
            t.total_ms = t.prep_ms + t.infer_ms + t.post_ms;
            t.post_timing = pt;
            return 0;
        };
        run_video_benchmark(infer, args, /*has_prep_stages=*/false, /*has_infer_stages=*/false);
    } else {
        RknnEngine engine;
        if (engine.init(args.model) != 0) {
            std::cerr << "❌ RKNN 初始化失败: " << args.model << std::endl;
            return 1;
        }
        const bool is_int8 = (args.backend == Backend::RknnInt8);
        engine.input_setting(is_int8 ? RKNN_TENSOR_INT8 : RKNN_TENSOR_FLOAT16);

        uint32_t in_h = 0, in_w = 0, channel = 0;
        if (!engine.get_input_hw_c(0, in_h, in_w, channel)) {
            std::cerr << "❌ 不支持的 RKNN 输入排布" << std::endl;
            return 1;
        }
        uint32_t field_count = 0, candidate_count = 0;
        if (!engine.get_output_layout(field_count, candidate_count)) {
            std::cerr << "❌ RKNN 输出维度不符合 YOLO 格式" << std::endl;
            return 1;
        }

        cv::VideoCapture probe(args.video);
        const int width  = probe.get(cv::CAP_PROP_FRAME_WIDTH);
        const int height = probe.get(cv::CAP_PROP_FRAME_HEIGHT);
        probe.release();
        preprocess_parameter = get_preprocess_parameter(width, height, in_w, in_h);

        std::vector<float> output_data;
        const float conf = args.conf, iou = args.iou;
        const int fcount = static_cast<int>(field_count);
        const int ccount = static_cast<int>(candidate_count);

        if (is_int8) {
            std::vector<int8_t> tensor_data(channel * in_h * in_w);
            InferFn infer = [&, conf, iou, fcount, ccount, ms](const cv::Mat& f, FrameTimings& t) -> int {
                const auto a = clock::now();
                PreprocessTiming ptt;
                preprocess_image(f, preprocess_parameter, tensor_data, TENSOR_NHWC, &ptt, Int8PackMode::Optimized);
                const auto b = clock::now();
                RknnRunTiming rt;
                const int ret = engine.run(tensor_data.data(), output_data, &rt);
                if (ret != 0) return ret;
                const auto c = clock::now();
                PostprocessTiming pt;
                postprocess_image(output_data.data(), fcount, ccount, conf, iou, preprocess_parameter, &pt);
                const auto d = clock::now();
                t.prep_ms = ms(a, b); t.infer_ms = ms(b, c); t.post_ms = ms(c, d);
                t.total_ms = t.prep_ms + t.infer_ms + t.post_ms;
                t.prep_timing = ptt; t.has_prep_stages = true;
                t.run_timing = rt;  t.has_infer_stages = true;
                t.post_timing = pt;
                return 0;
            };
            run_video_benchmark(infer, args, /*has_prep_stages=*/true, /*has_infer_stages=*/true);
        } else {
            std::vector<uint16_t> tensor_data(channel * in_h * in_w);
            InferFn infer = [&, conf, iou, fcount, ccount, ms](const cv::Mat& f, FrameTimings& t) -> int {
                const auto a = clock::now();
                preprocess_image(f, preprocess_parameter, tensor_data, TENSOR_NHWC);
                const auto b = clock::now();
                RknnRunTiming rt;
                const int ret = engine.run(tensor_data.data(), output_data, &rt);
                if (ret != 0) return ret;
                const auto c = clock::now();
                PostprocessTiming pt;
                postprocess_image(output_data.data(), fcount, ccount, conf, iou, preprocess_parameter, &pt);
                const auto d = clock::now();
                t.prep_ms = ms(a, b); t.infer_ms = ms(b, c); t.post_ms = ms(c, d);
                t.total_ms = t.prep_ms + t.infer_ms + t.post_ms;
                t.run_timing = rt; t.has_infer_stages = true;
                t.post_timing = pt;
                return 0;
            };
            run_video_benchmark(infer, args, /*has_prep_stages=*/false, /*has_infer_stages=*/true);
        }
    }
    return 0;
}
