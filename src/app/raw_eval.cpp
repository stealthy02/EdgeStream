// ============================================================================
// 【原始张量导出】C++ 多图原始输出评测 —— 不做后处理，直接保存模型原始输出
//
// 与 accuracy_eval 的区别：
//     accuracy_eval  : 推理 → 后处理(NMS/解码) → 保存检测框 → 比对框级指标
//     raw_eval       : 推理 → 原始输出张量逐元素写盘 → 比对张量级误差 (MAPE 等)
//
// 流水线（3 个后端，每张图各跑一次）：
//     ONNX (Ort C++)       → onnx/{stem}.bin
//     RKNN FP16 (librknnrt) → rknn_fp16/{stem}.bin
//     RKNN INT8 (librknnrt) → rknn_int8/{stem}.bin
//
// 每个后端目录额外写一个 manifest.json，记录每个 stem 的 shape / elem_num，
// 供 scripts/raw_eval_check.py 读取校验（MAPE / CosSim / MaxAbs / MeanAbs）。
//
// 输出目录（固定无日期，覆盖式写，可单独重跑单个后端）：
//     artifacts/raw_eval/
//       ├── onnx/      {stem}.bin + manifest.json
//       ├── rknn_fp16/ {stem}.bin + manifest.json
//       └── rknn_int8/ {stem}.bin + manifest.json
//
// 用法：
//   # 默认跑 32 张图（ONNX + RKNN FP16 + RKNN INT8）：
//   ./raw_eval --images-dir assets/regression/test_image
//
//   # 指定数量：
//   ./raw_eval --images-dir assets/regression/test_image --num-images 64
//
//   # 板端没装 RKNN 驱动？只导出 ONNX 原始输出：
//   ./raw_eval --images-dir assets/regression/test_image --onnx-only
//
//   # 导出完跑 Python 校验：
//   python3 scripts/raw_eval_check.py
// ============================================================================

#include "edgestream/core/file_io.h"
#include "edgestream/inference/onnx_engine.h"
#include "edgestream/inference/rknn_engine.h"
#include "edgestream/yolo/preprocess.h"
#include <opencv2/opencv.hpp>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace fs = std::filesystem;

// ============================================================================
// 原始输出保存器 —— 每个后端一个：写 {stem}.bin，收集 meta，最后统一写 manifest
// ============================================================================
struct BackendSaver
{
    std::string dir;            // 输出子目录，如 artifacts/raw_eval/onnx
    std::string backend;        // 后端名，如 onnx / rknn_fp16 / rknn_int8
    nlohmann::json outputs;     // stem → {file, shape, elem_num}
    int ok_cnt = 0;

    bool save(const std::string& stem,
              const std::vector<float>& data,
              const std::vector<int64_t>& shape)
    {
        std::ofstream f(dir + "/" + stem + ".bin", std::ios::binary);
        if (!f) return false;
        f.write(reinterpret_cast<const char*>(data.data()),
                static_cast<std::streamsize>(data.size() * sizeof(float)));
        f.close();
        if (!f.good()) return false;

        outputs[stem] = {
            {"file",     stem + ".bin"},
            {"shape",    shape},
            {"elem_num", data.size()},
        };
        ++ok_cnt;
        return true;
    }

    void write_manifest() const
    {
        nlohmann::json m;
        m["backend"] = backend;
        m["dtype"]   = "float32";
        m["count"]   = ok_cnt;
        m["outputs"] = outputs;
        std::ofstream f(dir + "/manifest.json");
        f << m.dump(2) << "\n";
    }
};

// ============================================================================
// ONNX 原始输出导出 —— 引擎外部传入，避免每次加载模型
// ============================================================================
static void onnx_infer_raw(
    OnnxEngine& engine,
    const cv::Mat& image,
    BackendSaver& saver,
    const std::string& stem)
{
    auto input_shape = engine.input_shape();
    PreprocessParameter pp = get_preprocess_parameter(
        image.cols, image.rows, input_shape[3], input_shape[2]
    );
    std::vector<float> tensor_data(3 * input_shape[2] * input_shape[3]);
    preprocess_image(image, pp, tensor_data, TENSOR_NCHW);

    auto output_list = engine.run(tensor_data);
    InferenceOutput merged = OnnxEngine::merge_split_outputs(std::move(output_list));

    if (saver.save(stem, merged.data, merged.shape)) {
        std::cout << "    onnx    → " << merged.data.size() << " floats";
    } else {
        std::cout << "    onnx    → ❌ 保存失败";
    }
}

// ============================================================================
// RKNN 原始输出导出 —— FP16 / INT8 由 quantization 参数区分
// ============================================================================
static void rknn_infer_raw(
    RknnEngine& engine,
    const cv::Mat& image,
    BackendSaver& saver,
    const std::string& stem,
    bool quantization)
{
    uint32_t in_h = 0, in_w = 0, ch = 0;
    if (!engine.get_input_hw_c(0, in_h, in_w, ch)) {
        std::cerr << "❌ 不支持的 RKNN 输入排布" << std::endl;
        return;
    }

    PreprocessParameter pp = get_preprocess_parameter(image.cols, image.rows, in_w, in_h);
    std::vector<float> output_data;
    int run_ret;

    if (quantization) {
        std::vector<int8_t> tensor_data(ch * in_h * in_w);
        preprocess_image(image, pp, tensor_data, TENSOR_NHWC);
        engine.input_setting(RKNN_TENSOR_INT8);
        run_ret = engine.run(tensor_data.data(), output_data);
    } else {
        std::vector<uint16_t> tensor_data(ch * in_h * in_w);
        preprocess_image(image, pp, tensor_data, TENSOR_NHWC);
        engine.input_setting(RKNN_TENSOR_FLOAT16);
        run_ret = engine.run(tensor_data.data(), output_data);
    }
    if (run_ret != 0) {
        std::cerr << " ❌ RKNN 推理失败 ret=" << run_ret << std::endl;
        return;
    }

    // run() 内部已按 [boxes(4), scores(C)] 拼接 split 输出，布局等价 [1, 84, 8400]，
    // 与 ONNX merge_split_outputs 的结果逐元素对齐。
    std::vector<int64_t> shape;
    uint32_t field_count = 0, candidate_count = 0;
    if (engine.get_output_layout(field_count, candidate_count)) {
        shape = {1, static_cast<int64_t>(field_count),
                    static_cast<int64_t>(candidate_count)};
    } else {
        shape = {1, static_cast<int64_t>(output_data.size())};  // 兜底：平铺一维
    }

    const char* tag = quantization ? "rknn_i8" : "rknn_fp";
    if (saver.save(stem, output_data, shape)) {
        std::cout << "    " << tag << " → " << output_data.size() << " floats";
    } else {
        std::cout << "    " << tag << " → ❌ 保存失败";
    }
}

// ============================================================================
// 扫描图片目录，返回有序 stem→full_path
// ============================================================================
static std::vector<std::pair<std::string, std::string>> scan_image_dir(const std::string& dir)
{
    std::vector<std::pair<std::string, std::string>> result;
    static const std::set<std::string> exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"};
    if (!fs::is_directory(dir)) {
        std::cerr << "❌ 图片目录不存在: " << dir << std::endl;
        return result;
    }
    for (const auto& entry : fs::directory_iterator(dir)) {
        if (!entry.is_regular_file()) continue;
        std::string ext = entry.path().extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(),
                       [](unsigned char c){ return std::tolower(c); });
        if (exts.count(ext)) {
            result.emplace_back(entry.path().stem().string(),
                                entry.path().string());
        }
    }
    std::sort(result.begin(), result.end(),
              [](const auto& a, const auto& b){ return a.first < b.first; });
    return result;
}

// ============================================================================
// 参数解析（轻量手动版，不引 boost）
// ============================================================================
struct Args {
    std::string output_root = "artifacts/raw_eval";
    std::string images_dir  = "assets/regression/test_image";
    std::string onnx_model  = "models/onnx/yolo11s_640_split.onnx";
    std::string rknn_fp16   = "models/rknn/yolo11s_640_split.rknn";
    std::string rknn_int8   = "models/rknn/yolo11s_640_split_int8.rknn";
    int  num_images = 32;
    bool run_onnx = true;
    bool run_rknn = true;
};

static void print_usage(const char* prog)
{
    std::cerr
        << "用法: " << prog << " [选项]\n\n"
        << "  --output-root <DIR>   输出根目录 (默认: artifacts/raw_eval)\n"
        << "                        结果按后端写入 {root}/onnx|rknn_fp16|rknn_int8/{stem}.bin\n"
        << "  --images-dir <DIR>    测试图片目录 (默认: assets/regression/test_image)\n"
        << "  --num-images <N>      测试图片数量 (默认: 32)\n"
        << "  --onnx-model <PATH>   ONNX split 模型路径\n"
        << "  --rknn-fp16 <PATH>    RKNN FP16 模型路径\n"
        << "  --rknn-int8 <PATH>    RKNN INT8 模型路径\n"
        << "  --onnx-only           只导出 ONNX 原始输出 (跳过 RKNN)\n"
        << "  --rknn-only           只导出 RKNN 原始输出 (跳过 ONNX)\n"
        << "  -h / --help           显示此帮助\n";
}

static Args parse_args(int argc, char* argv[])
{
    Args a;
    for (int i = 1; i < argc; ++i) {
        std::string s = argv[i];
        auto need_next = [&](const char* name) -> std::string {
            if (i + 1 >= argc) {
                throw std::runtime_error(std::string("参数 ") + name + " 缺少值");
            }
            return std::string(argv[++i]);
        };
        if (s == "-h" || s == "--help")        { print_usage(argv[0]); exit(0); }
        else if (s == "--output-root")         { a.output_root = need_next("--output-root"); }
        else if (s == "--images-dir")          { a.images_dir  = need_next("--images-dir"); }
        else if (s == "--num-images" || s == "-n") {
            a.num_images = std::stoi(need_next("--num-images"));
            if (a.num_images <= 0) throw std::runtime_error("--num-images 必须为正整数");
        }
        else if (s == "--onnx-model")          { a.onnx_model  = need_next("--onnx-model"); }
        else if (s == "--rknn-fp16")           { a.rknn_fp16   = need_next("--rknn-fp16"); }
        else if (s == "--rknn-int8")           { a.rknn_int8   = need_next("--rknn-int8"); }
        else if (s == "--onnx-only")           { a.run_onnx = true;  a.run_rknn = false; }
        else if (s == "--rknn-only")           { a.run_onnx = false; a.run_rknn = true; }
        else {
            std::cerr << "⚠️  未知参数: " << s << "\n\n";
            print_usage(argv[0]);
            throw std::runtime_error("参数解析失败");
        }
    }
    return a;
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

    if (!mkdir_if_not_exist(args.output_root)) {
        std::cerr << "❌ output_root 不存在且无法创建: " << args.output_root << std::endl;
        return 1;
    }
    // 按后端建子目录 + 准备保存器
    // 注意：保存器必须是具名对象。不能用 vector 存放后再取元素指针——
    // push_back 扩容搬移元素会使指针悬空，后续读 dir 得到垃圾字符串，
    // 在 string 拼接处抛 std::length_error。
    BackendSaver onnx_saver, fp_saver, i8_saver;
    auto init_saver = [&](BackendSaver& s, const char* name) {
        s.backend = name;
        s.dir = args.output_root + "/" + name;
        mkdir_if_not_exist(s.dir);
    };
    if (args.run_onnx) init_saver(onnx_saver, "onnx");
    if (args.run_rknn) {
        init_saver(fp_saver, "rknn_fp16");
        init_saver(i8_saver, "rknn_int8");
    }

    std::cout << "========================================" << std::endl;
    std::cout << "【原始张量导出】raw_eval（无后处理）" << std::endl;
    std::cout << "  output_root = " << args.output_root << std::endl;
    std::cout << "  images    = " << args.images_dir << " (前 " << args.num_images << " 张)" << std::endl;
    std::cout << "  后端      = "
              << (args.run_onnx ? "ONNX " : "")
              << (args.run_rknn ? "RKNN-FP16 RKNN-INT8 " : "") << std::endl;
    std::cout << "========================================" << std::endl;

    // 扫描图片，截取前 num_images 张
    auto images = scan_image_dir(args.images_dir);
    if (images.empty()) {
        std::cerr << "❌ 图片目录里没有找到可用图片: " << args.images_dir << std::endl;
        return 1;
    }
    if (static_cast<int>(images.size()) > args.num_images) {
        images.resize(static_cast<std::size_t>(args.num_images));
    }
    std::cout << "[信息] 本次导出 " << images.size() << " 张图片\n" << std::endl;

    // —— ONNX 引擎：只加载一次，循环复用 ——
    std::unique_ptr<OnnxEngine> onnx_engine;
    if (args.run_onnx) {
        if (!fs::exists(args.onnx_model)) {
            std::cerr << "❌ ONNX 模型不存在: " << args.onnx_model << std::endl;
            return 1;
        }
        std::cout << "[加载] ONNX 模型: " << args.onnx_model << std::endl;
        onnx_engine = std::make_unique<OnnxEngine>(args.onnx_model);
    }
    // —— RKNN 引擎：FP16 / INT8 各加载一次，循环复用 ——
    std::unique_ptr<RknnEngine> rknn_fp_engine;
    std::unique_ptr<RknnEngine> rknn_i8_engine;
    if (args.run_rknn) {
        for (const auto& p : {args.rknn_fp16, args.rknn_int8}) {
            if (!fs::exists(p)) {
                std::cerr << "❌ RKNN 模型不存在: " << p << std::endl;
                return 1;
            }
        }
        std::cout << "[加载] RKNN FP16 模型: " << args.rknn_fp16 << std::endl;
        rknn_fp_engine = std::make_unique<RknnEngine>();
        if (rknn_fp_engine->init(args.rknn_fp16) != 0) {
            std::cerr << "❌ RKNN FP16 初始化失败: " << args.rknn_fp16 << std::endl;
            return 1;
        }
        std::cout << "[加载] RKNN INT8 模型: " << args.rknn_int8 << std::endl;
        rknn_i8_engine = std::make_unique<RknnEngine>();
        if (rknn_i8_engine->init(args.rknn_int8) != 0) {
            std::cerr << "❌ RKNN INT8 初始化失败: " << args.rknn_int8 << std::endl;
            return 1;
        }
        std::cout << std::endl;
    }

    // —— 逐图循环：推理 → 原始张量直接写盘 ——
    for (std::size_t i = 0; i < images.size(); ++i) {
        const auto& stem = images[i].first;
        const auto& img_path = images[i].second;
        std::cout << "[" << (i + 1) << "/" << images.size() << "] "
                  << stem << " (" << fs::path(img_path).filename().string() << ")" << std::endl;

        cv::Mat image = cv::imread(img_path, cv::IMREAD_COLOR);
        if (image.empty()) {
            std::cout << "  ❌ 无法读取图片，跳过" << std::endl;
            continue;
        }

        if (args.run_onnx) {
            onnx_infer_raw(*onnx_engine, image, onnx_saver, stem);
            std::cout << std::endl;
        }
        if (args.run_rknn) {
            rknn_infer_raw(*rknn_fp_engine, image, fp_saver, stem, /*quant=*/false);
            std::cout << std::endl;
            rknn_infer_raw(*rknn_i8_engine, image, i8_saver, stem, /*quant=*/true);
            std::cout << std::endl;
        }
    }

    // —— 每个后端写 manifest.json（shape / elem_num 元数据）——
    if (args.run_onnx) onnx_saver.write_manifest();
    if (args.run_rknn) {
        fp_saver.write_manifest();
        i8_saver.write_manifest();
    }

    std::cout << "\n========================================" << std::endl;
    std::cout << "✅ 全部完成，共导出 " << images.size() << " 张图的原始输出" << std::endl;
    if (args.run_onnx) std::cout << "   onnx      : " << onnx_saver.ok_cnt << " 个 .bin" << std::endl;
    if (args.run_rknn) {
        std::cout << "   rknn_fp16 : " << fp_saver.ok_cnt << " 个 .bin" << std::endl;
        std::cout << "   rknn_int8 : " << i8_saver.ok_cnt << " 个 .bin" << std::endl;
    }
    std::cout << "📁 输出根目录: " << args.output_root << std::endl;
    std::cout << "▶  下一步跑 Python 校验 (MAPE):" << std::endl;
    std::cout << "   python3 scripts/raw_eval_check.py" << std::endl;
    return 0;
}
